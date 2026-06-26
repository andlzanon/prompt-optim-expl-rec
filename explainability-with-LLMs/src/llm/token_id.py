def get_token():
    """
    Return the access token string used by the LLM-loading code.

    This helper exposes the token value stored in this module so other parts
    of the codebase can pass it to external model-loading utilities that
    require authenticated access.

    Parameters
    ----------
    None
        This function does not receive any parameters.

    Returns
    -------
    str
        Access token string defined inside the function body.
    """

    token_access = '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'
    return token_access