from src.metrics.etd import etd_metric
from src.metrics.sep import sep_metric

from typing import Dict, List, Any
import pandas as pd

def parse_graph_explanations(
    explanation_blocks: List[str],
) -> Dict[str, List[str]]:
    """
    Parse explanation blocks and extract graph components from each valid line.

    Each input block is expected to be a multiline string in which every
    relevant line follows the explanation-path format used by the project. The
    function collects the interacted item and the middle entity (the attribute
    node) for every line that can be parsed successfully.

    Parameters
    ----------
    explanation_blocks : List[str]
        List of multiline explanation strings. Non-string entries are ignored.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with two keys:
        - ``"middle_entities"``: attribute nodes extracted from the middle of
          each valid explanation path.
        - ``"interacted_items"``: interacted items extracted from the first
          node after the ``|`` separator.

    Raises
    ------
    None directly.
        Invalid or malformed entries are skipped instead of raising errors.

    Notes
    -----
    This helper is part of the metrics pipeline. It converts textual
    explanation paths into the graph components consumed by ETD and SEP
    scoring.

    Expected line format:
        <Rec> | <Interacted> -> <Attr> -> <Rec>
    """

    middle_entities: List[str] = []
    interacted_items: List[str] = []

    for expl_block in explanation_blocks:
        if not isinstance(expl_block, str):
            continue

        for expl in expl_block.splitlines():
            expl = expl.strip()
            if not expl or "|" not in expl or "->" not in expl:
                continue

            # Ignore the repeated recommended-item prefix and keep only the
            # actual path structure used by the graph-based metrics.
            _, rest = expl.split("|", 1)
            parts = [p.strip() for p in rest.split("->")]

            # A valid path must at least expose interacted item, attribute, and
            # recommended item in that order.
            if len(parts) < 3:
                continue

            interacted_items.append(parts[0])
            middle_entities.append(parts[1])

    return {
        "middle_entities": middle_entities,
        "interacted_items": interacted_items,
    }

def score_etd_from_explanations(
    user_explanations: Dict[Any, str],
    total_types: int,
) -> float:
    """
    Compute ETD from textual explanation blocks and average the score by user.

    The function parses each user's explanation block into middle entities and
    forwards those entity labels to ``etd_metric``. The final result is the
    arithmetic mean of the per-user ETD scores computed from valid string
    blocks.

    Parameters
    ----------
    user_explanations : Dict[Any, str]
        Mapping from user identifier to a multiline explanation string.
        Entries whose value is not a string are ignored.
    total_types : int
        Total number of distinct explanation types expected by the ETD metric.

    Returns
    -------
    float
        Mean ETD score across processed users. Returns ``0.0`` when no valid
        string explanation blocks are available.

    Raises
    ------
    Exception
        Any exception raised by ``etd_metric`` may propagate.

    Notes
    -----
    This function is a bridge between the text-based explainability outputs and
    the graph-based ETD evaluation used in the optimization flow.
    """

    scores: List[float] = []

    for _, block in user_explanations.items():
        if not isinstance(block, str):
            continue

        parsed = parse_graph_explanations([block])
        ents = [e for e in parsed["middle_entities"] if e]

        scores.append(float(etd_metric(
            explanation_types=ents,
            total_types=total_types
        )))

    return float(sum(scores) / max(1, len(scores)))

def score_sep_from_explanations(
    user_explanations: Dict[Any, str],
    props_df: pd.DataFrame,
    memo_sep: dict,
    beta: float = 0.3,
) -> float:
    """
    Compute SEP from textual explanation blocks and average the score by user.

    The function parses each user's explanation block into middle entities,
    converts each entity into the singleton-list structure expected by
    ``sep_metric``, and averages the resulting per-user SEP scores.

    Parameters
    ----------
    user_explanations : Dict[Any, str]
        Mapping from user identifier to a multiline explanation string.
        Entries whose value is not a string are ignored.
    props_df : pd.DataFrame
        DataFrame passed directly to ``sep_metric`` as ``prop_set``.
    memo_sep : dict
        Mutable memoization dictionary forwarded to ``sep_metric``.
    beta : float, default=0.3
        Exponential decay parameter forwarded to ``sep_metric``.

    Returns
    -------
    float
        Mean SEP score across processed users. Returns ``0.0`` when no valid
        string explanation blocks are available.

    Raises
    ------
    Exception
        Any exception raised by ``sep_metric`` may propagate.
        
    Notes
    -----
    This function converts textual explanation paths into the minimal property
    structure required by the SEP metric used during prompt optimization.
    """

    scores: List[float] = []

    for _, block in user_explanations.items():
        if not isinstance(block, str):
            continue

        parsed = parse_graph_explanations([block])
        ents = [e for e in parsed["middle_entities"] if e]
        # sep_metric expects each explanation as a list of properties.
        props = [[e] for e in ents]

        scores.append(float(sep_metric(
            beta=beta,
            props=props,
            prop_set=props_df,
            memo_sep=memo_sep
        )))

    return float(sum(scores) / max(1, len(scores)))

def combine_sep_etd_f1(
    sep_value: float,
    etd_value: float,
) -> float:
    """
    Combine SEP and ETD with a symmetric harmonic mean.

    Parameters
    ----------
    sep_value : float
        SEP score in the ``[0, 1]`` range.
    etd_value : float
        ETD score in the ``[0, 1]`` range.

    Returns
    -------
    float
        Harmonic-mean combination of the two scores.
    """

    sep_value = float(sep_value)
    etd_value = float(etd_value)
    if sep_value <= 0.0 and etd_value <= 0.0:
        return 0.0
    return float((2.0 * sep_value * etd_value) / (sep_value + etd_value))

def score_graph_metrics_from_explanations(
    user_explanations: Dict[Any, str],
    props_df: pd.DataFrame,
    memo_sep: dict,
    total_types: int,
    beta: float = 0.3,
) -> Dict[str, float]:
    """
    Compute the graph-based explainability scores used by this project.

    Parameters
    ----------
    user_explanations : Dict[Any, str]
        Mapping from user identifier to the multiline explanation block
        generated for that user.
    props_df : pd.DataFrame
        Knowledge-graph dataframe consumed by SEP.
    memo_sep : dict
        Mutable memoization dictionary forwarded to SEP.
    total_types : int
        Total number of distinct explanation types available in the dataset.
    beta : float, default=0.3
        Exponential decay parameter forwarded to SEP.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the standalone ``sep`` and ``etd`` scores and
        the symmetric combined score ``sep_etd_f1``.
    """

    sep_value = float(
        score_sep_from_explanations(
            user_explanations=user_explanations,
            props_df=props_df,
            memo_sep=memo_sep,
            beta=beta,
        )
    )
    etd_value = float(
        score_etd_from_explanations(
            user_explanations=user_explanations,
            total_types=total_types,
        )
    )

    return {
        "sep": sep_value,
        "etd": etd_value,
        "sep_etd_f1": combine_sep_etd_f1(sep_value=sep_value, etd_value=etd_value),
    }
