"""Shared constants for the JLC Flux2 ControlNet integration."""

PROJECT_LOG_PREFIX = "[JLC Flux2 ControlNet]"
REQUESTS_KEY = "jlc_flux2_controlnet_requests"
WRAPPER_KEY = "jlc_flux2_controlnet.execute_only"
CONTROL_LAYERS = (0, 2, 4, 6)
EXPECTED_CONTROL_INPUT_CHANNELS = 260
EXPECTED_FLUX2_LATENT_CHANNELS = 128
EXPECTED_FLUX2_PATCH_SIZE = 1
EXPECTED_MASK_CHANNELS = 4
