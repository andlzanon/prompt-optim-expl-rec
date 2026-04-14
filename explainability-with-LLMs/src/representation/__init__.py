from src.representation.base_representation import BaseRepresentation

def available_representations() -> tuple[str, ...]:
    return ("sbert", "llm2vec")


def build_representation(name: str, **kwargs) -> BaseRepresentation:
    normalized_name = str(name).strip().lower()

    if normalized_name == "sbert":
        from src.representation.sbert import SBERTRepresentation

        return SBERTRepresentation(**kwargs)

    if normalized_name == "llm2vec":
        from src.representation.l2v import LLM2VecRepresentation

        return LLM2VecRepresentation(**kwargs)

    options = ", ".join(available_representations())
    raise ValueError(f"Unknown representation '{name}'. Available options: {options}.")
