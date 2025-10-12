import click


def validate_sigma_parameters(
    sigma: float | None, sigma_tangent: float | None, sigma_radial: float | None
) -> None:
    """
    Validate the sigma parameters for the unsharp mask command.
    You must provide either a single --sigma value or both --sigma-tangent and --sigma-radial.

    :raises click.BadParameter: If the parameters are not valid.
    """
    s = sigma is not None
    st = sigma_tangent is not None
    sr = sigma_radial is not None
    if (s and not (st or sr)) or (not s and (st and sr)):
        return

    raise click.BadParameter(
        "You must provide either a single sigma value or both --sigma-tangent and --sigma-radial"
    )
