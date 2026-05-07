from src.llm.token_id import get_token

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import re
import torch
import pandas as pd

MODEL_ID = {
    "Llama3.1-I": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Llama3.1": "meta-llama/Meta-Llama-3.1-8B",
}

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

class LLM:
    """
    Wrapper around a causal LLM used to select explanation paths.

    This class loads a chat-oriented language model, builds the prompts used
    for path selection, queries the model for one option number, validates the
    response format, and converts the final choices into a tabular explanation
    output consumed by the rest of the project.
    """

    _YEAR_RE = re.compile(r"\s*\(\d{4}\)$")
    _IDX_RE = re.compile(r"\b(\d{1,6})\b")

    def __init__(self, llm_method: str = "", seed: int = 2026):
        """
        Initialize the LLM wrapper and its runtime configuration.

        Parameters
        ----------
        llm_method : str, default=""
            Key used to resolve the Hugging Face model identifier from
            ``MODEL_ID``.
        seed : int, default=2026
            Seed value stored on the instance and later reused by helper
            methods such as random path sampling.

        Returns
        -------
        None
            This constructor initializes the instance in place.

        Notes
        -----
        Model weights are not loaded here. They are loaded later by
        ``set_model()``.
        """

        self.seed = seed
        self.llm_method = llm_method
        if self.llm_method:
            if self.llm_method not in MODEL_ID:
                raise KeyError(
                    f"Unknown llm_method {self.llm_method!r}. "
                    f"Expected one of: {sorted(MODEL_ID.keys())}"
                )
            self.model_name = MODEL_ID[self.llm_method]
        else:
            self.model_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fixed_system_prompt = None
        self.metric_selection_guidance = None
        self.system_prompt = None
        self.prompt = None
        self.model = None
        self.tokenizer = None
        self.terminators = None
        self.token_access = None
        self.last_selected_paths_df = pd.DataFrame()

    def refresh_system_prompt(self) -> None:
        """
        Rebuild the final system prompt from its fixed and optimizable parts.

        Returns
        -------
        None
            This method updates instance attributes in place.

        Side Effects
        ------------
        Mutates ``self.system_prompt`` and ``self.prompt``.
        """

        self.system_prompt = self.fixed_system_prompt + self.metric_selection_guidance
        self.prompt = [{"role": "system", "content": self.system_prompt}]

    def set_prompt(self) -> None:
        """
        Build the system prompt used for explanation-path selection.

        The method defines the instruction that constrains the model to return
        exactly one option number and stores both the raw system prompt text
        and the initial chat message structure expected by later prompt
        builders.

        Returns
        -------
        None
            This method updates instance attributes and does not return a
            value.

        Side Effects
        ------------
        Mutates ``self.system_prompt`` and ``self.prompt``.

        Notes
        -----
        This prompt forms the foundation for every model request performed by
        the explainability flow.
        """

        self.fixed_system_prompt = (
            "You are selecting ONE explanation path for the given recommendation.\n"
            "Return ONLY the final answer.\n"
            "Output MUST be exactly ONE token: the OPTION NUMBER.\n"
            "No extra text.\n"
            "Rules:\n"
            "- Choose EXACTLY ONE of the provided numbered paths.\n"
            "- Do NOT invent items, attributes, or paths.\n"
            "- Do NOT output words, punctuation, or explanations.\n"
            "- Output ONLY the integer corresponding to the chosen option.\n"
        )
        self.metric_selection_guidance = (
            "Selection criteria (priority order):\n"
            "1) Prefer attributes that give the most informative, specific, and discriminative explanation.\n"
            "2) Avoid overly generic attributes when more descriptive ones are available.\n"
            "3) If tied, prefer the option whose attribute most clearly connects the two items in the path.\n"
        )
        self.refresh_system_prompt()

    def set_model(self) -> None:
        """
        Load the causal language model, tokenizer, and termination tokens.

        The method initializes the system prompt, retrieves the authentication
        token, loads the configured Hugging Face model with 4-bit
        quantization, loads the tokenizer, and stores the token ids that mark
        the end of generation.

        Returns
        -------
        None
            This method updates the instance with loaded model components.

        Raises
        ------
        Exception
            Any exception raised by token retrieval, model loading, tokenizer
            loading, or token-id conversion may propagate.

        Side Effects
        ------------
        Loads model weights into memory and mutates ``self.model``,
        ``self.tokenizer``, ``self.terminators``, ``self.token_access``,
        ``self.system_prompt``, and ``self.prompt``.

        Notes
        -----
        This is the main setup step required before the instance can generate
        explanations.
        """

        if self.model_name is None:
            raise ValueError(
                "Cannot load the model because llm_method was not provided when "
                "initializing LLM."
            )

        self.set_prompt()
        self.token_access = get_token()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device,
            offload_buffers=True,
            token=self.token_access,
            trust_remote_code=True,
            use_cache=True,
            quantization_config=quantization_config,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.token_access,
            use_safetensors=True,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    @classmethod
    def remove_year_from_title(cls, title: str) -> str:
        """
        Remove a trailing year suffix from a title string.

        Parameters
        ----------
        title : str
            Title text that may end with a year pattern such as ``"(1999)"``.

        Returns
        -------
        str
            Title without the trailing year suffix. Non-string inputs are
            returned unchanged.

        Raises
        ------
        None directly.

        Notes
        -----
        This helper is used to normalize movie titles before prompt creation
        and explanation formatting.
        """

        if not isinstance(title, str):
            return title
        return cls._YEAR_RE.sub("", title).strip()

    def sample_paths_randomly(self, paths_df: pd.DataFrame, num_paths: int) -> pd.DataFrame:
        """
        Select a random subset of explanation paths from a DataFrame.

        Parameters
        ----------
        paths_df : pd.DataFrame
            Candidate explanation paths.
        num_paths : int
            Number of rows to sample.

        Returns
        -------
        pd.DataFrame
            Original DataFrame when ``num_paths`` is greater than or equal to
            its length; otherwise a randomly sampled subset using ``self.seed``
            as ``random_state``.

        Raises
        ------
        ValueError
            May be propagated by pandas sampling if the input is malformed.

        Notes
        -----
        This helper currently provides the path-selection behavior used by
        ``generate_explanations``.
        """

        if num_paths >= len(paths_df):
            return paths_df
        return paths_df.sample(n=num_paths, random_state=self.seed)

    def _load_user_paths_df(self, user_paths_path: str) -> pd.DataFrame:
        """
        Load and normalize one user's explanation-path table.

        Parameters
        ----------
        user_paths_path : str
            CSV path containing the candidate explanation paths for one user.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame with normalized item-title columns when present.
        """

        user_paths_df = pd.read_csv(user_paths_path)

        if "recommended_item_name" in user_paths_df.columns:
            user_paths_df["recommended_item_name"] = user_paths_df[
                "recommended_item_name"
            ].apply(self.remove_year_from_title)
        if "interacted_item_name" in user_paths_df.columns:
            user_paths_df["interacted_item_name"] = user_paths_df[
                "interacted_item_name"
            ].apply(self.remove_year_from_title)

        return user_paths_df

    def _select_paths_for_recommendation(
        self,
        paths_for_rec_df: pd.DataFrame,
        selection_strategy: str,
        num_paths_per_recommendation: int,
    ) -> pd.DataFrame:
        """
        Select the candidate paths used for one recommendation.

        Notes
        -----
        The current implementation still routes every strategy to random
        sampling. The strategy name is preserved for logging and future
        strategy-specific branches.
        """

        if selection_strategy == "random":
            return self.sample_paths_randomly(
                paths_for_rec_df,
                num_paths_per_recommendation,
            )

        return self.sample_paths_randomly(
            paths_for_rec_df,
            num_paths_per_recommendation,
        )

    @staticmethod
    def _build_valid_lines_list(paths_df: pd.DataFrame) -> list[str]:
        """
        Convert explanation-path rows into formatted text lines.

        Parameters
        ----------
        paths_df : pd.DataFrame
            DataFrame expected to contain ``interacted_item_name``,
            ``recommended_item_name``, and ``common_props`` columns.

        Returns
        -------
        list[str]
            Formatted explanation strings in the form
            ``<Rec> | <Interacted> -> <Attr> -> <Rec>``.

        Raises
        ------
        KeyError
            May be raised if required columns are missing.

        Notes
        -----
        These formatted lines are later used both as prompt options and as the
        final persisted explanation text.
        """
        
        return [
            f"{r_name} | {i_name} -> {prop} -> {r_name}"
            for i_name, r_name, prop in zip(
                paths_df["interacted_item_name"],
                paths_df["recommended_item_name"],
                paths_df["common_props"],
            )
        ]

    def _build_selected_paths_snapshot(
        self,
        selected_paths_df: pd.DataFrame,
        user_id: int,
        recommendation_order: int,
        selection_strategy: str,
        source_paths_path: str,
        available_paths_count: int,
    ) -> pd.DataFrame:
        """
        Build a tabular snapshot of the paths sampled for one recommendation.

        Parameters
        ----------
        selected_paths_df : pd.DataFrame
            DataFrame containing only the candidate paths sampled for the
            current user and recommendation.
        user_id : int
            Identifier of the user whose paths were sampled.
        recommendation_order : int
            One-based order of the processed recommendation for that user.
        selection_strategy : str
            Strategy label used to choose the candidate paths.
        source_paths_path : str
            Source CSV file from which the candidate paths were loaded.
        available_paths_count : int
            Total number of candidate paths available for the recommendation
            before sampling.

        Returns
        -------
        pd.DataFrame
            Snapshot DataFrame enriched with user, selection, and ordering
            metadata for later persistence.
        """

        snapshot_df = selected_paths_df.copy().reset_index(drop=True)
        snapshot_df.insert(0, "userId", user_id)
        snapshot_df["recommendation_order"] = recommendation_order
        snapshot_df["selection_strategy"] = selection_strategy
        snapshot_df["selection_seed"] = self.seed
        snapshot_df["available_paths_for_recommendation"] = available_paths_count
        snapshot_df["selected_paths_for_recommendation"] = len(snapshot_df)
        snapshot_df["selection_order"] = range(1, len(snapshot_df) + 1)
        snapshot_df["source_paths_file"] = source_paths_path
        snapshot_df["selected_path"] = self._build_valid_lines_list(snapshot_df)
        return snapshot_df

    @staticmethod
    def _selected_paths_required_columns() -> list[str]:
        """
        Return the required schema used by preselected-path CSV files.
        """

        return [
            "userId",
            "recommendation_order",
            "recommended_item_id",
            "interacted_item_id",
            "selection_strategy",
            "selection_seed",
            "selection_order",
            "available_paths_for_recommendation",
            "selected_paths_for_recommendation",
            "common_props",
            "interacted_item_name",
            "recommended_item_name",
            "selected_path",
            "source_paths_file",
        ]

    def _normalize_selected_paths_df(
        self,
        selected_paths_df: pd.DataFrame,
        users: list[int],
    ) -> pd.DataFrame:
        """
        Validate and normalize a precomputed selected-paths DataFrame.
        """

        if not isinstance(selected_paths_df, pd.DataFrame):
            raise TypeError("selected_paths_df must be a pandas DataFrame.")

        required_cols = set(self._selected_paths_required_columns())
        missing = required_cols - set(selected_paths_df.columns)
        if missing:
            raise ValueError(
                "Missing required columns in selected_paths_df: "
                f"{sorted(list(missing))}"
            )

        normalized_df = selected_paths_df.copy()
        normalized_df = normalized_df[normalized_df["userId"].isin(users)].copy()

        if "recommended_item_name" in normalized_df.columns:
            normalized_df["recommended_item_name"] = normalized_df[
                "recommended_item_name"
            ].apply(self.remove_year_from_title)
        if "interacted_item_name" in normalized_df.columns:
            normalized_df["interacted_item_name"] = normalized_df[
                "interacted_item_name"
            ].apply(self.remove_year_from_title)

        normalized_df = normalized_df.sort_values(
            ["userId", "recommendation_order", "selection_order", "recommended_item_id"]
        ).reset_index(drop=True)

        return normalized_df

    def prepare_selected_paths(
        self,
        users: list[int],
        explanation_paths_prefix: str,
        selection_strategy: str = "random",
        num_recommendations: int = 3,
        num_paths_per_recommendation: int = 3,
    ) -> pd.DataFrame:
        """
        Precompute and return the candidate paths sampled for the requested users.

        Returns
        -------
        pd.DataFrame
            Tabular snapshot containing the sampled candidate paths for every
            processed user and recommendation.
        """

        selected_paths_rows: list[dict] = []

        for user_id in tqdm(users, desc="Preparing selected paths", ascii=True):
            user_paths_path = f"{explanation_paths_prefix}_{user_id}_user_id.csv"
            if not os.path.exists(user_paths_path):
                raise FileNotFoundError(
                    f"Explanation paths file not found: {user_paths_path}"
                )

            user_paths_df = self._load_user_paths_df(user_paths_path)

            recommended_item_ids = user_paths_df["recommended_item_id"].drop_duplicates()
            if num_recommendations > len(recommended_item_ids):
                raise ValueError(
                    f"Requested {num_recommendations} recommendations, but only "
                    f"{len(recommended_item_ids)} available for user {user_id}."
                )

            selected_recommended_ids = recommended_item_ids.head(num_recommendations)

            for recommendation_order, rec_id in enumerate(
                selected_recommended_ids,
                start=1,
            ):
                paths_for_rec_df = user_paths_df[
                    user_paths_df["recommended_item_id"] == rec_id
                ]
                selected_paths_df = self._select_paths_for_recommendation(
                    paths_for_rec_df=paths_for_rec_df,
                    selection_strategy=selection_strategy,
                    num_paths_per_recommendation=num_paths_per_recommendation,
                )
                selected_paths_snapshot = self._build_selected_paths_snapshot(
                    selected_paths_df=selected_paths_df,
                    user_id=user_id,
                    recommendation_order=recommendation_order,
                    selection_strategy=selection_strategy,
                    source_paths_path=user_paths_path,
                    available_paths_count=len(paths_for_rec_df),
                )
                selected_paths_rows.extend(
                    selected_paths_snapshot.to_dict(orient="records")
                )

        self.last_selected_paths_df = pd.DataFrame(selected_paths_rows)
        return self.last_selected_paths_df

    @classmethod
    def _try_parse_choice_index(cls, text: str, k: int) -> int | None:
        """
        Extract a valid option index from the model output text.

        Parameters
        ----------
        text : str
            Raw model output to inspect.
        k : int
            Maximum valid option number.

        Returns
        -------
        int | None
            Parsed integer option when the text contains a number in the
            inclusive range ``[1, k]``; otherwise ``None``.

        Raises
        ------
        None directly.

        Notes
        -----
        This helper is used by the retry loop that enforces one-number
        outputs from the model.
        """

        if not isinstance(text, str):
            return None
        m = cls._IDX_RE.search(text.strip())
        if not m:
            return None
        idx = int(m.group(1))
        return idx if 1 <= idx <= k else None

    @staticmethod
    def _extract_attribute_from_output(text: str) -> str | None:
        """
        Extract the middle attribute node from a formatted explanation string.

        Parameters
        ----------
        text : str
            Explanation string expected in the form
            ``<Rec> | <Interacted> -> <Attr> -> <Rec>``.

        Returns
        -------
        str | None
            Extracted attribute text when the input matches the expected
            structure; otherwise ``None``.

        Raises
        ------
        None directly.

        Notes
        -----
        The extracted attribute is used to track attribute reuse across a
        user's explanations.
        """

        # <Rec> | <Interacted> -> <Attr> -> <Rec>
        if not isinstance(text, str) or " | " not in text:
            return None
        rhs = text.split(" | ", 1)[1].strip()
        parts = rhs.split(" -> ")
        if len(parts) != 3:
            return None
        return parts[1].strip() or None

    def build_explanation_path_selection_user_message_single_rec(
        self,
        paths_df: pd.DataFrame,
        user_history_df: pd.DataFrame,
        include_user_history: bool = True,
        used_attributes: list[str] | None = None,
        max_history_items: int = 200,
    ) -> str:
        """
        Build the user-facing message for selecting one path for one item.

        Parameters
        ----------
        paths_df : pd.DataFrame
            DataFrame containing the candidate paths for a single recommended
            item.
        user_history_df : pd.DataFrame
            DataFrame containing the user's interaction history.
        include_user_history : bool, default=True
            Whether recent user history should be included in the message when
            available.
        used_attributes : list[str] | None, default=None
            Previously selected attributes for the current user. When provided,
            they are listed as contextual information.
        max_history_items : int, default=200
            Maximum number of recent interaction titles shown when history is
            included.

        Returns
        -------
        str
            Prompt text presented as the user message in the chat request.

        Raises
        ------
        KeyError
            May be raised if required columns are missing from the provided
            DataFrames.
        IndexError
            May be raised if ``paths_df`` is empty because the code accesses
            the first recommended item name.

        Notes
        -----
        This method constructs the main per-recommendation prompt content used
        by the model to choose one explanation path.
        """

        chunks: list[str] = []

        # Optional user history
        if include_user_history and user_history_df is not None and not user_history_df.empty:
            uh = user_history_df.sort_values(by="timestamp", ascending=False).head(max_history_items)

            chunks.append(
                "Context: a user has interacted with some items. "
                "The most recent interacted items are listed below:\n"
            )

            chunks.extend(
                [f"{i}. {t}\n" for i, t in enumerate(uh["title"].tolist(), start=1)]
            )

            chunks.append("\n")

        else:
            
            chunks.append(
                "Task: select ONE explanation path for the recommendation below.\n\n"
            )

        rec_name = paths_df["recommended_item_name"].iloc[0]

        chunks.append(
            f"Recommended item: '{rec_name}'.\n"
            "Each option is an explanation path connecting an interacted item to the recommended item via one attribute.\n"
            "The arrow '->' indicates the connection between an item and an attribute.\n\n"
        )

        chunks.append("Options:\n")

        for i, (i_name, r_name, prop) in enumerate(
            zip(
                paths_df["interacted_item_name"],
                paths_df["recommended_item_name"],
                paths_df["common_props"],
            ),
            start=1,
        ):
            chunks.append(f"{i}. {i_name} -> {prop} -> {r_name}\n")

        chunks.append("\n")

        # SEP-friendly guidance
        chunks.append(
            "Selection guidance:\n"
            "- Prefer attributes that provide more informative, specific, and discriminative explanations.\n"
            "- Avoid attributes that are overly broad or apply to many items when a more specific attribute exists.\n"
            "- Prefer attributes that better explain why the recommended item is related to the interacted item.\n\n"
        )

        # Optional context about previously used attributes
        if used_attributes:
            counts = Counter(used_attributes)

            chunks.append("Attributes already used in explanations for this user (context only):\n")

            for attr, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                chunks.append(f"- {attr}: {count}\n")

            chunks.append("\n")

        return "".join(chunks)

    def create_prompt_for_explainability_single_message(self, user_message: str) -> list[dict]:
        """
        Combine the system prompt with one user message into a chat payload.

        Parameters
        ----------
        user_message : str
            User message describing the recommendation and candidate paths.

        Returns
        -------
        list[dict]
            Two-message chat prompt containing the stored system prompt and the
            provided user message.

        Raises
        ------
        TypeError
            May occur later if ``self.prompt`` is not in the expected format.

        Notes
        -----
        This helper prepares the structure consumed by the tokenizer chat
        template before generation.
        """

        return [self.prompt[0], {"role": "user", "content": user_message}]

    @torch.inference_mode()
    def request_model(self, prompt: list[dict], k: int) -> str:
        """
        Request one model completion for a path-selection prompt.

        Parameters
        ----------
        prompt : list[dict]
            Chat-style prompt passed to the tokenizer chat template.
        k : int
            Number of candidate options available. It is used to define a
            small upper bound for ``max_new_tokens``.

        Returns
        -------
        str
            Decoded model output text after removing special tokens.

        Raises
        ------
        Exception
            Any exception raised during tokenization, device transfer,
            generation, or decoding may propagate.

        Side Effects
        ------------
        Runs model generation in inference mode on the loaded model.

        Notes
        -----
        The generation length is intentionally small because the expected
        output is only one option number.
        """

        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        max_new = max(2, len(str(k)) + 1)

        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            use_cache=True,
        )

        gen = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def request_model_with_validation_and_retry(
        self,
        prompt: list[dict],
        k: int,
        max_retries: int = 1,
    ) -> tuple[int | None, int, bool, str]:
        """
        Request a model choice, validate it, and optionally retry on failure.

        Parameters
        ----------
        prompt : list[dict]
            Initial chat-style prompt.
        k : int
            Number of valid answer options.
        max_retries : int, default=1
            Maximum number of correction retries after the initial attempt.

        Returns
        -------
        tuple[int | None, int, bool, str]
            Tuple containing:
            - parsed choice index or ``None``
            - number of attempts performed
            - whether a valid choice was obtained
            - raw model output from the last attempt

        Raises
        ------
        Exception
            Any exception raised by ``request_model`` may propagate.

        Notes
        -----
        When the model output is invalid, the method appends an explicit
        correction message and tries again until the retry budget is exhausted.
        """

        tries = 0
        raw = ""
        local_prompt = prompt

        while tries <= max_retries:
            tries += 1
            raw = self.request_model(local_prompt, k=k).strip()

            idx = self._try_parse_choice_index(raw, k)
            if idx is not None:
                return idx, tries, True, raw

            correction = (
                "INVALID.\n"
                f"Output ONLY ONE integer token from 1 to {k}.\n"
                f"Previous output: {raw}\n"
                "Now output ONLY the number:"
            )
            local_prompt = local_prompt + [{"role": "user", "content": correction}]

        return None, tries, False, raw

    def generate_explanations(
        self,
        users: list[int],
        interactions_df: pd.DataFrame,
        explanation_paths_prefix: str,
        selection_strategy: str = "random",
        num_recommendations: int = 3,
        num_paths_per_recommendation: int = 3,
        selected_paths_df: pd.DataFrame | None = None,
        include_user_history: bool = True,
        max_retries: int = 1,
        max_history_items: int = 200,
        empty_cache_every: int = 0,  # 0 = never
    ) -> pd.DataFrame:
        """
        Generate explanations for multiple users and recommended items.

        The method loads each user's explanation-path file, builds prompts for
        a subset of recommended items, asks the LLM to choose one path per
        item, validates the response, and returns the collected results as a
        DataFrame.

        Parameters
        ----------
        users : list[int]
            User identifiers to process.
        interactions_df : pd.DataFrame
            Interaction history used to provide user context and titles.
        explanation_paths_prefix : str
            Prefix used to build each user's explanation-path CSV file path.
        selection_strategy : str, default="random"
            Strategy name used to choose candidate paths. In the current
            implementation, all values route to random sampling.
        num_recommendations : int, default=3
            Number of recommended items processed per user.
        num_paths_per_recommendation : int, default=3
            Number of candidate paths sampled for each recommended item.
        include_user_history : bool, default=True
            Whether recent user history is included in the prompt when
            available.
        max_retries : int, default=1
            Maximum number of retries used by the response-validation loop.
        max_history_items : int, default=200
            Maximum number of user-history items included in each prompt.
        empty_cache_every : int, default=0
            Frequency for calling ``torch.cuda.empty_cache()``. When ``0``,
            cache clearing is disabled.

        Returns
        -------
        pd.DataFrame
            DataFrame containing one row per user/recommended-item pair with
            the selected explanation, validation metadata, and raw model
            output.

        Raises
        ------
        ValueError
            Raised when a user has no interaction history in
            ``interactions_df`` or when the requested number of recommendations
            exceeds what is available for a user.
        FileNotFoundError
            Raised when the expected explanation-path CSV file for a user does
            not exist.
        Exception
            Any exception raised by pandas operations or by the model request
            helpers may propagate.

        Side Effects
        ------------
        Reads per-user CSV files from disk, performs model generation, and may
        clear the CUDA cache periodically.

        Notes
        -----
        This method is the main explainability-generation entry point used by
        both the standalone explainability run and the prompt-optimization
        workflow.
        """

        rows: list[dict] = []

        interactions_df = interactions_df.copy()
        if "title" in interactions_df.columns:
            interactions_df["title"] = interactions_df["title"].apply(self.remove_year_from_title)

        if selected_paths_df is None:
            selected_paths_source_df = self.prepare_selected_paths(
                users=users,
                explanation_paths_prefix=explanation_paths_prefix,
                selection_strategy=selection_strategy,
                num_recommendations=num_recommendations,
                num_paths_per_recommendation=num_paths_per_recommendation,
            )
        else:
            selected_paths_source_df = self._normalize_selected_paths_df(
                selected_paths_df=selected_paths_df,
                users=users,
            )

        self.last_selected_paths_df = selected_paths_source_df.copy()

        for u_i, user_id in enumerate(tqdm(users, desc="Generating explanations", ascii=True), start=1):
            user_history_df = interactions_df[interactions_df["userId"] == user_id]
            if user_history_df.empty:
                raise ValueError(f"User {user_id} has no interaction history in interactions_df.")

            user_selected_paths_df = selected_paths_source_df[
                selected_paths_source_df["userId"] == user_id
            ].copy()
            if user_selected_paths_df.empty:
                raise ValueError(
                    f"No selected candidate paths were found for user {user_id}."
                )

            recommendation_pairs = (
                user_selected_paths_df[["recommendation_order", "recommended_item_id"]]
                .drop_duplicates()
                .sort_values(["recommendation_order", "recommended_item_id"])
            )
            if len(recommendation_pairs) != num_recommendations:
                raise ValueError(
                    f"Expected {num_recommendations} prepared recommendations for "
                    f"user {user_id}, but found {len(recommendation_pairs)}. "
                    "Check whether the precomputed selected-paths CSV was "
                    "generated with the same configuration."
                )

            used_attributes: list[str] = []

            for recommendation_order, rec_id in recommendation_pairs.itertuples(index=False):
                selected_paths_df = (
                    user_selected_paths_df[
                        (user_selected_paths_df["recommendation_order"] == recommendation_order)
                        & (user_selected_paths_df["recommended_item_id"] == rec_id)
                    ]
                    .sort_values("selection_order")
                    .reset_index(drop=True)
                )

                valid_lines = self._build_valid_lines_list(selected_paths_df)
                k = len(valid_lines)

                user_message = self.build_explanation_path_selection_user_message_single_rec(
                    paths_df=selected_paths_df,
                    user_history_df=user_history_df,
                    include_user_history=include_user_history,
                    used_attributes=used_attributes,
                    max_history_items=max_history_items,
                )
                prompt = self.create_prompt_for_explainability_single_message(user_message)

                chosen_idx, tries, is_valid, raw = self.request_model_with_validation_and_retry(
                    prompt=prompt, k=k, max_retries=max_retries
                )

                if is_valid and chosen_idx is not None:
                    explanation = valid_lines[chosen_idx - 1]
                    attr = self._extract_attribute_from_output(explanation)
                    if attr:
                        used_attributes.append(attr)
                else:
                    explanation = "-"

                rows.append(
                    {
                        "userId": user_id,
                        "recommended_item_id": rec_id,
                        "explanation": explanation,
                        "tries": tries,
                        "valid": is_valid,
                        "raw_model_output": raw,
                    }
                )

            if empty_cache_every and (u_i % empty_cache_every == 0):
                torch.cuda.empty_cache()

        return pd.DataFrame(rows)
