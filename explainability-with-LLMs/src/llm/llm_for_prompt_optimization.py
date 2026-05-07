from src.representation.embedding_utils.mmr import mmr_select
from src.llm.llm_for_explainability import LLM
from src.representation import build_representation
from src.representation.base_representation import BaseRepresentation

from typing import List, Dict, Callable, Tuple, Any, Optional
import os
import json
import time
import numpy as np
import torch
import pandas as pd

class PromptOptimizer:
    """
    Coordinate iterative prompt optimization for explanation-path selection.

    This class manages the loop that evaluates prompts on training and
    validation users, generates new candidate system instructions from prior
    high-performing prompts, and persists per-epoch artifacts and summary
    metadata.
    """

    def __init__(
        self,
        epochs: int = 10,
        meta_prompt_instruction_quantity: int = 3,
        eval_every: int = 1,
        patience: int = 3,
        min_delta: float = 1e-2,
        early_stopping: bool = False,
        save_dir: str = "out/prompt_opt",
        mmr_lambda_quality: float = 1.0,   # 0..1 (higher = more relevance, less diversity)
        mmr_pool_multiplier: int = 10,     # candidate pool size = K * multiplier
        representation_model: str = "llm2vec",
        representation: Optional[BaseRepresentation] = None,
        representation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the prompt-optimization controller and its dependencies.

        Parameters
        ----------
        epochs : int, default=10
            Maximum number of optimization epochs to run.
        meta_prompt_instruction_quantity : int, default=3
            Number of previously ranked prompts used as references when
            generating a new prompt.
        eval_every : int, default=1
            Validation frequency in epochs. Validation runs when
            ``(epoch + 1) % eval_every == 0`` and ``eval_every > 0``.
        patience : int, default=3
            Number of consecutive non-improving validation checks tolerated by
            the early-stopping logic.
        min_delta : float, default=1e-2
            Minimum validation improvement considered meaningful for early
            stopping.
        early_stopping : bool, default=False
            Whether validation-based early stopping should be enabled.
        save_dir : str, default="out/prompt_opt"
            Directory where epoch artifacts and metadata are stored.
        mmr_lambda_quality : float, default=1.0
            Relevance-versus-diversity trade-off used by MMR when selecting
            reference prompts.
        mmr_pool_multiplier : int, default=10
            Multiplier controlling how many top-ranked prompts are considered
            before MMR selection.
        representation_model : str, default="llm2vec"
            Name of the representation backend to build when
            ``representation`` is not provided.
        representation : Optional[BaseRepresentation], default=None
            Prebuilt representation instance. When provided, it is used
            directly instead of constructing one from ``representation_model``.
        representation_kwargs : Optional[Dict[str, Any]], default=None
            Keyword arguments forwarded to ``build_representation`` when a
            representation instance must be created.

        Returns
        -------
        None
            This constructor initializes the optimizer instance in place.

        Raises
        ------
        Exception
            Any exception raised while constructing the representation backend
            or creating the save directory may propagate.

        Side Effects
        ------------
        May instantiate a representation model and creates ``save_dir`` if it
        does not already exist.

        Notes
        -----
        The initializer also defines the meta-prompt template and generation
        configuration used later in the optimization loop.
        """

        self.epochs = epochs
        self.meta_prompt_instruction_quantity = meta_prompt_instruction_quantity
        self.eval_every = eval_every
        self.patience = patience
        self.min_delta = min_delta
        self.early_stopping = bool(early_stopping)

        self.mmr_lambda_quality = float(mmr_lambda_quality)
        self.mmr_pool_multiplier = int(mmr_pool_multiplier)
        self.representation_model = str(representation_model).lower()
        self.representation_kwargs = representation_kwargs or {}
        self.representation = representation or build_representation(
            self.representation_model,
            **self.representation_kwargs,
        )
        self.embedding_cache: Dict[str, np.ndarray] = {}

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # The optimizer rewrites only the selection-guidance block, which is
        # the portion of the final system prompt that belongs to the search
        # space.
        self.meta_prompt_template = """\
You generate ONE new candidate selection-guidance block for an explanation-path selection task in a recommender system.

TASK CONTEXT (what the user message contains):
- For ONE recommended item, a list of K numbered explanation paths (1..K), each in the form:
  <Interacted Item> -> <Attribute> -> <Recommended Item>

REFERENCE GUIDANCE BLOCKS (do NOT copy; use only for inspiration):
- The examples below were selected from previous high-performing guidance blocks and are shown from BEST to WORST by training score.
{instructions}

SELECTION CRITERIA (must be included; in priority order):
1) Prefer attributes that make the explanation more informative, specific, and discriminative.
2) Avoid overly generic attributes when more descriptive ones are available.
3) If tied, prefer the option whose attribute most clearly connects the two items in the path.

STYLE REQUIREMENTS:
- Write only the selection-guidance block, not a full system prompt.
- Start with a short heading or label for the guidance block.
- Use imperative language and explicit constraints when useful.
- Avoid single-sentence guidance.
- Different wording/structure than the reference.

Return ONLY the new selection-guidance block text (no quotes, no markdown).
"""

        self.gen_cfg = {
            "max_new_tokens": 500,
            "temperature": 1.2,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 6,
        }

    def _epoch_dir(self, epoch: int) -> str:
        """
        Build and create the directory used to store artifacts for one epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index used in the folder name.

        Returns
        -------
        str
            Path to the epoch-specific directory.

        Raises
        ------
        OSError
            May be propagated if the directory cannot be created.

        Side Effects
        ------------
        Creates the epoch directory on disk when it does not exist.
        """
        
        d = os.path.join(self.save_dir, f"epoch_{epoch:03d}")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_json(self, path: str, data: dict) -> None:
        """
        Persist a dictionary to disk as a formatted JSON file.

        Parameters
        ----------
        path : str
            Destination JSON path.
        data : dict
            Dictionary to serialize.

        Returns
        -------
        None
            This helper writes the file and does not return a value.

        Raises
        ------
        OSError
            May be propagated if the file cannot be written.
        TypeError
            May be raised if ``data`` contains non-serializable values.

        Side Effects
        ------------
        Creates or overwrites the target file.
        """

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _json_safe(self, value: Any) -> Any:
        """
        Convert nested values into a structure suitable for JSON serialization.

        Parameters
        ----------
        value : Any
            Input value that may contain nested dictionaries, sequences, or
            other Python objects.

        Returns
        -------
        Any
            JSON-friendlier version of ``value`` where dictionary keys are
            stringified, sequences are converted recursively, primitive values
            are preserved, and unsupported objects become strings.

        Raises
        ------
        None directly.
            The method performs only local type conversion.

        Notes
        -----
        This helper is used when storing run settings and metadata that may
        contain objects not directly serializable by ``json.dump``.
        """

        if isinstance(value, dict):
            return {str(key): self._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _df_to_explanation_blocks(
        self,
        df: pd.DataFrame,
        user_col: str = "userId",
        explanation_col: str = "explanation",
    ) -> Dict[Any, str]:
        """
        Convert an explanations DataFrame into per-user multiline text blocks.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the user and explanation columns.
        user_col : str, default="userId"
            Column used to group explanations by user.
        explanation_col : str, default="explanation"
            Column containing the explanation-path text.

        Returns
        -------
        Dict[Any, str]
            Mapping from user identifier to a multiline string containing that
            user's valid explanations joined with newline characters. Returns an
            empty dictionary when the input DataFrame is ``None`` or empty.

        Raises
        ------
        KeyError
            May be raised if ``df`` does not contain the requested columns.

        Notes
        -----
        This helper prepares generated explanations for the graph-based metric
        functions, which expect user-to-block mappings instead of raw
        DataFrames.
        """

        if df is None or df.empty:
            return {}

        tmp = df[[user_col, explanation_col]].dropna().copy()
        tmp[explanation_col] = tmp[explanation_col].astype(str)

        tmp = tmp[tmp[explanation_col].str.contains(r"\|", regex=True)]
        tmp = tmp[tmp[explanation_col].str.contains("->", regex=False)]

        grouped = tmp.groupby(user_col)[explanation_col].apply(lambda s: "\n".join(s.tolist()))
        return grouped.to_dict()

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """
        Normalize a list of text inputs into plain strings.

        Parameters
        ----------
        texts : List[str]
            Input texts that may contain ``None`` values or non-string items.

        Returns
        -------
        List[str]
            List in which ``None`` values become empty strings and all other
            items are converted with ``str(...)``.

        Raises
        ------
        None directly.

        Notes
        -----
        This helper standardizes text inputs before embedding generation.
        """

        return ["" if text is None else str(text) for text in texts]

    def _encode_texts_with_cache(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts with the configured representation while reusing a cache.

        Parameters
        ----------
        texts : List[str]
            Texts to encode.

        Returns
        -------
        np.ndarray
            Matrix of embeddings aligned with the input order. Returns an empty
            ``float32`` array with shape ``(0, 0)`` when ``texts`` is empty
            after preparation.

        Raises
        ------
        ValueError
            Raised when the representation backend returns a number of
            embeddings that does not match the number of missing texts.
        Exception
            Any exception raised by the representation backend may propagate.

        Side Effects
        ------------
        Populates ``self.embedding_cache`` with newly encoded texts.

        Notes
        -----
        This cache is used to avoid recomputing prompt embeddings during MMR
        reference selection across epochs.
        """

        prepared_texts = self._prepare_texts(texts)
        if not prepared_texts:
            return np.empty((0, 0), dtype=np.float32)

        # Preserve the first occurrence order while requesting each unseen text
        # from the representation backend only once.
        missing_texts = list(
            dict.fromkeys(text for text in prepared_texts if text not in self.embedding_cache)
        )
        if missing_texts:
            missing_embeddings = self.representation.encode(missing_texts)
            if missing_embeddings.shape[0] != len(missing_texts):
                raise ValueError("Representation model returned an unexpected number of embeddings.")
            for text, embedding in zip(missing_texts, missing_embeddings):
                self.embedding_cache[text] = np.asarray(embedding, dtype=np.float32)

        return np.vstack([self.embedding_cache[text] for text in prepared_texts]).astype(
            np.float32,
            copy=False,
        )

    def _select_mmr_refs(
        self,
        ranked: List[Tuple[str, float, int]],
        k: int,
    ) -> List[Tuple[str, float, int]]:
        """
        Select reference prompts from ranked history using MMR diversification.

        Parameters
        ----------
        ranked : List[Tuple[str, float, int]]
            Ranked prompt history where each entry is
            ``(prompt_text, train_score, epoch_index)``.
        k : int
            Maximum number of reference prompts to return.

        Returns
        -------
        List[Tuple[str, float, int]]
            Selected ranked entries ordered according to the index sequence
            returned by ``mmr_select``. Returns an empty list when ``ranked`` is
            empty or ``k`` is non-positive.

        Raises
        ------
        ValueError
            May propagate from ``mmr_select`` when MMR parameters are invalid.
        Exception
            May propagate from the representation backend used to encode
            prompts.

        Notes
        -----
        The candidate pool is first restricted to the highest-scoring prompts,
        then MMR is used to keep a balance between quality and diversity.
        """

        if not ranked or k <= 0:
            return []

        ranked_sorted = sorted(ranked, key=lambda x: x[1], reverse=True)
        pool_size = min(len(ranked_sorted), max(k, k * self.mmr_pool_multiplier))
        pool = ranked_sorted[:pool_size]
        if not pool:
            return []

        candidate_embeddings = self._encode_texts_with_cache([prompt for prompt, _, _ in pool])
        query_embedding = candidate_embeddings[0:1]

        selected_indexes = mmr_select(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            top_k=k,
            lambda_param=self.mmr_lambda_quality,
        )
        return [pool[index] for index in selected_indexes]

    def _meta_prompt_used(self, examples_block: str) -> str:
        """
        Fill the meta-prompt template with the selected reference examples.

        Parameters
        ----------
        examples_block : str
            Formatted block containing the reference prompts shown to the model.

        Returns
        -------
        str
            Final meta-prompt text produced from the class template.
        """

        return self.meta_prompt_template.format(instructions=examples_block)

    def _create_meta_messages(self, meta_prompt_used: str) -> List[Dict[str, str]]:
        """
        Build the chat message list used to request a new prompt instruction.

        Parameters
        ----------
        meta_prompt_used : str
            Meta-prompt text that becomes the system message.

        Returns
        -------
        List[Dict[str, str]]
            Two-message chat payload containing one system message and one user
            request asking for a new system instruction.
        """

        return [
            {"role": "system", "content": meta_prompt_used},
            {
                "role": "user",
                "content": "Return one new selection-guidance block now. Output only the block text.",
            },
        ]

    def _generate_one_instruction(self, llm: LLM, meta_prompt_used: str) -> Tuple[str, float]:
        """
        Generate one new candidate system instruction from the meta-prompt.

        Parameters
        ----------
        llm : LLM
            LLM wrapper that provides the tokenizer, model, and terminator ids
            used for generation.
        meta_prompt_used : str
            Final meta-prompt text used to condition generation.

        Returns
        -------
        Tuple[str, float]
            Tuple ``(prompt_text, generation_time_seconds)``.

        Raises
        ------
        Exception
            Any exception raised during tokenization, device transfer,
            generation, decoding, or cache clearing may propagate.

        Side Effects
        ------------
        Runs model generation and calls ``torch.cuda.empty_cache()`` after the
        generation step.

        Notes
        -----
        The input batch is explicitly moved to ``"cuda"`` in the current
        implementation.
        """

        msgs = self._create_meta_messages(meta_prompt_used)

        inputs = llm.tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        t0 = time.time()
        outputs = llm.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.gen_cfg["max_new_tokens"],
            eos_token_id=llm.terminators,
            pad_token_id=llm.tokenizer.eos_token_id,
            do_sample=True,
            temperature=self.gen_cfg["temperature"],
            top_p=self.gen_cfg["top_p"],
            top_k=self.gen_cfg["top_k"],
            repetition_penalty=self.gen_cfg["repetition_penalty"],
            no_repeat_ngram_size=self.gen_cfg["no_repeat_ngram_size"],
            use_cache=False,
        )
        gen_time = float(time.time() - t0)

        resp = outputs[0][inputs["input_ids"].shape[-1] :]
        prompt = llm.tokenizer.decode(resp, skip_special_tokens=True).strip()

        torch.cuda.empty_cache()
        return prompt, gen_time

    def run_optimize_process(
        self,
        llm: LLM,
        metric_fn: Callable[[Any], float],
        train_user_ids,
        val_user_ids,
        interactions_df_train,
        interactions_df_val,
        explain_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Tuple[str, float, int]]]:
        """
        Run the full prompt-optimization loop and collect its artifacts.

        The method iteratively evaluates prompts on the training users,
        optionally evaluates them on validation users, uses MMR-selected past
        prompts to generate new candidate instructions, and records both
        per-epoch artifacts and summary metadata.

        Parameters
        ----------
        llm : LLM
            LLM instance used to generate explanations and new prompt
            instructions.
        metric_fn : Callable[[Any], float]
            Metric callable that scores a mapping of user explanations.
        train_user_ids : Any
            Collection of user identifiers used for training evaluation.
        val_user_ids : Any
            Collection of user identifiers used for validation evaluation.
        interactions_df_train : Any
            Interaction data passed to ``llm.generate_explanations`` for the
            training users.
        interactions_df_val : Any
            Interaction data passed to ``llm.generate_explanations`` for the
            validation users.
        explain_kwargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments forwarded to
            ``llm.generate_explanations``.

        Returns
        -------
        Tuple[Dict[str, Any], List[Tuple[str, float, int]]]
            Tuple ``(info, ranked_train)`` where ``info`` contains run
            metadata, epoch history, and best-prompt summaries, and
            ``ranked_train`` contains the training-ranked prompts as
            ``(prompt_text, train_score, epoch_index)`` tuples.

        Raises
        ------
        Exception
            Any exception raised by directory creation, explanation generation,
            metric evaluation, prompt generation, or JSON/CSV persistence may
            propagate.

        Side Effects
        ------------
        Creates per-epoch directories, writes ``epoch.json`` files, writes
        training explanation CSVs, writes validation CSVs when validation runs,
        mutates the passed ``llm`` instance by updating ``system_prompt`` and
        ``prompt``, and prints per-epoch progress lines.

        Notes
        -----
        The best final selection is tracked by training metric. When early
        stopping is enabled, validation results are compared against the
        previous validation run.
        """

        explain_kwargs = explain_kwargs or {}

        ranked_train: List[Tuple[str, float, int]] = []
        info: Dict[str, Any] = {
            "baseline_prompt": llm.system_prompt,
            "baseline_metric_selection_guidance": llm.metric_selection_guidance,
            "final_meta_prompt_template": self.meta_prompt_template,
            "settings": {
                "epochs": self.epochs,
                "meta_prompt_instruction_quantity": self.meta_prompt_instruction_quantity,
                "eval_every": self.eval_every,
                "patience": self.patience,
                "min_delta": self.min_delta,
                "early_stopping": self.early_stopping,
                "mmr_lambda_quality": self.mmr_lambda_quality,
                "mmr_pool_multiplier": self.mmr_pool_multiplier,
                "representation_model": self.representation_model,
                "representation_kwargs": self._json_safe(self.representation_kwargs),
            },
            "epochs_history": [],
        }

        best_train_metric = float("-inf")
        best_train_prompt: Optional[str] = None
        best_train_system_prompt: Optional[str] = None
        best_train_epoch: Optional[int] = None

        best_val_metric = float("-inf")
        best_val_prompt: Optional[str] = None
        best_val_system_prompt: Optional[str] = None
        best_val_epoch: Optional[int] = None

        no_improve = 0
        prev_val_metric: Optional[float] = None

        for epoch in range(self.epochs):
            ep_dir = self._epoch_dir(epoch)

            # Epoch zero evaluates the baseline guidance already stored in the
            # LLM instance; later epochs generate new guidance from references.
            if epoch == 0:
                guidance_this_epoch = llm.metric_selection_guidance
                gen_time = 0.0
                generated = False
                meta_prompt_used = None
                selected_refs: List[Tuple[str, float, int]] = []
            else:
                selected_refs = self._select_mmr_refs(
                    ranked=ranked_train,
                    k=self.meta_prompt_instruction_quantity,
                )
                selected_refs = sorted(selected_refs, key=lambda x: x[1], reverse=True)

                blocks = []
                for j, (p, s, e) in enumerate(selected_refs, start=1):
                    blocks.append(
                        f"Example {j} (train_score={s:.6f}, epoch={e})\n---\n{p.strip()}\n---\n"
                    )
                examples_block = "\n".join(blocks)
                meta_prompt_used = self._meta_prompt_used(examples_block)

                guidance_this_epoch, gen_time = self._generate_one_instruction(llm, meta_prompt_used)
                generated = True

            # Update only the optimizable guidance block, then rebuild the
            # final system prompt used during explanation generation.
            llm.metric_selection_guidance = guidance_this_epoch
            llm.refresh_system_prompt()

            epoch_json: Dict[str, Any] = {
                "epoch": epoch,
                "prompt": {
                    "guidance_this_epoch": guidance_this_epoch,
                    "prompt_this_epoch": llm.system_prompt,
                    "system_prompt_this_epoch": llm.system_prompt,
                    "generated_new_prompt": generated,
                    "time_spent_instruction": float(gen_time),
                },
                "artifacts": {
                    "epoch_json": "epoch.json",
                    "train_csv": "train_explanations.csv",
                    "val_csv": "val_explanations.csv",
                },
                "early_stopping_enabled": self.early_stopping,
                "early_stopping_actually_stopped": False,
            }

            if meta_prompt_used is not None:
                epoch_json["prompt"]["meta_prompt_used"] = meta_prompt_used

            # ---- TRAIN ----
            t0 = time.time()
            df_train = llm.generate_explanations(
                users=train_user_ids,
                interactions_df=interactions_df_train,
                **explain_kwargs,
            )
            t_train = float(time.time() - t0)

            train_blocks = self._df_to_explanation_blocks(df_train)
            train_metric = float(metric_fn(train_blocks))

            df_train.to_csv(os.path.join(ep_dir, "train_explanations.csv"), index=False)

            epoch_json["train_eval"] = {
                "train_metric": float(train_metric),
                "time_spent_train_eval": float(t_train),
                "train_rows": int(df_train.shape[0]),
                "train_valid_rate": float(df_train["valid"].mean())
                if "valid" in df_train.columns and len(df_train) > 0
                else None,
            }

            if train_metric > best_train_metric:
                best_train_metric = train_metric
                best_train_prompt = guidance_this_epoch
                best_train_system_prompt = llm.system_prompt
                best_train_epoch = epoch

            ranked_train.append((guidance_this_epoch, train_metric, epoch))
            ranked_train.sort(key=lambda x: x[1], reverse=True)

            print(f"\n[EPOCH {epoch}] TRAIN={train_metric:.6f} (gen={gen_time:.2f}s, train={t_train:.2f}s)")

            # ---- VAL + OPTIONAL EARLY STOPPING ----
            ran_val = False
            val_metric: Optional[float] = None
            t_val: Optional[float] = None
            val_improvement_vs_prev: Optional[float] = None
            prev_val_before_update: Optional[float] = None
            early_stopping_triggered_here = False

            if self.eval_every > 0 and ((epoch + 1) % self.eval_every == 0):
                ran_val = True

                t0 = time.time()
                df_val = llm.generate_explanations(
                    users=val_user_ids,
                    interactions_df=interactions_df_val,
                    **explain_kwargs,
                )
                t_val = float(time.time() - t0)

                val_blocks = self._df_to_explanation_blocks(df_val)
                val_metric = float(metric_fn(val_blocks))

                df_val.to_csv(os.path.join(ep_dir, "val_explanations.csv"), index=False)

                print(f"[EPOCH {epoch}] VAL={val_metric:.6f}")

                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_val_prompt = guidance_this_epoch
                    best_val_system_prompt = llm.system_prompt
                    best_val_epoch = epoch

                prev_val_before_update = prev_val_metric
                if self.early_stopping and prev_val_metric is not None:
                    val_improvement_vs_prev = float(val_metric - prev_val_metric)

                    # "improved" means: delta >= min_delta
                    if val_improvement_vs_prev < self.min_delta:
                        no_improve += 1
                        if no_improve >= self.patience:
                            early_stopping_triggered_here = True
                            epoch_json["early_stopping_triggered_here"] = True
                            info["early_stopping"] = {
                                "stopped_at_epoch": epoch,
                                "reason": "val_no_improvement_vs_previous",
                                "prev_val_metric": float(prev_val_metric),
                                "curr_val_metric": float(val_metric),
                                "improvement": float(val_improvement_vs_prev),
                                "min_delta": float(self.min_delta),
                                "patience": int(self.patience),
                                "eval_every": int(self.eval_every),
                                "no_improve_streak": int(no_improve),
                            }
                    else:
                        no_improve = 0

                prev_val_metric = float(val_metric)

                epoch_json["val_eval"] = {
                    "val_metric": float(val_metric),
                    "time_spent_val_eval": float(t_val),
                    "val_rows": int(df_val.shape[0]),
                    "val_valid_rate": float(df_val["valid"].mean())
                    if "valid" in df_val.columns and len(df_val) > 0
                    else None,
                    "prev_val_metric": float(prev_val_before_update) if prev_val_before_update is not None else None,
                    "val_improvement_vs_prev": float(val_improvement_vs_prev) if val_improvement_vs_prev is not None else None,
                    "no_improve_streak": int(no_improve) if self.early_stopping else None,
                }

            epoch_json["val_ran_this_epoch"] = ran_val
            epoch_json["early_stopping_triggered_here"] = bool(early_stopping_triggered_here)
            if early_stopping_triggered_here:
                epoch_json["early_stopping_actually_stopped"] = True

            self._save_json(os.path.join(ep_dir, "epoch.json"), epoch_json)

            refs_debug = []
            if selected_refs:
                for p, s, e in selected_refs:
                    refs_debug.append({"epoch": int(e), "train_score": float(s)})

            info["epochs_history"].append(
                {
                    "epoch": epoch,
                    "epoch_dir": ep_dir,
                    "generated_new_prompt": generated,
                    "time_spent_instruction": float(gen_time),
                    "train_metric": float(train_metric),
                    "time_spent_train_eval": float(t_train),
                    "train_rows": int(df_train.shape[0]),
                    "train_valid_rate": float(df_train["valid"].mean())
                    if "valid" in df_train.columns and len(df_train) > 0
                    else None,
                    "val_ran_this_epoch": ran_val,
                    "val_metric": float(val_metric) if val_metric is not None else None,
                    "time_spent_val_eval": float(t_val) if t_val is not None else None,
                    "prev_val_metric": float(prev_val_before_update) if prev_val_before_update is not None else None,
                    "val_improvement_vs_prev": float(val_improvement_vs_prev) if val_improvement_vs_prev is not None else None,
                    "no_improve_streak": int(no_improve) if self.early_stopping else None,
                    "early_stopping_triggered_here": bool(early_stopping_triggered_here),
                    "artifacts": {
                        "epoch_json_path": os.path.join(ep_dir, "epoch.json"),
                        "train_csv_path": os.path.join(ep_dir, "train_explanations.csv"),
                        "val_csv_path": os.path.join(ep_dir, "val_explanations.csv") if ran_val else None,
                    },
                    "meta_prompt_used": meta_prompt_used,
                    "guidance_this_epoch": guidance_this_epoch,
                    "prompt_this_epoch": llm.system_prompt,
                    "system_prompt_this_epoch": llm.system_prompt,
                    "mmr_selected_reference_epochs": refs_debug if refs_debug else None,
                }
            )

            if early_stopping_triggered_here:
                break

        info["best_on_train"] = {
            "best_train_metric": float(best_train_metric),
            "best_train_epoch": int(best_train_epoch) if best_train_epoch is not None else None,
            "best_train_prompt": best_train_prompt,
            "best_train_system_prompt": best_train_system_prompt,
        }
        info["best_on_validation"] = {
            "best_val_metric": float(best_val_metric),
            "best_val_epoch": int(best_val_epoch) if best_val_epoch is not None else None,
            "best_val_prompt": best_val_prompt,
            "best_val_system_prompt": best_val_system_prompt,
        }
        info["run_summary"] = {
            "epochs_completed": int(len(info["epochs_history"])),
            "early_stopped": "early_stopping" in info,
            "embedding_cache_size": int(len(self.embedding_cache)),
        }

        return info, ranked_train
