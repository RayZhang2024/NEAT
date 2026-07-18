"""Securely configure secrets for a laptop-hosted NEAT assistant server."""

from __future__ import annotations

import argparse
import getpass
import secrets

from tools.assistant_shared_server import (
    SERVER_ACCESS_TOKEN_NAME,
    SERVER_KEYRING_SERVICE_NAME,
    SERVER_OPENAI_KEY_NAME,
    load_server_credential,
)


def _keyring():
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError(
            "Secure setup requires the assistant dependency 'keyring'. "
            "Install it with: python -m pip install -e \".[assistant,assistant-server]\""
        ) from exc
    return keyring


def configure() -> str:
    """Prompt for the OpenAI key and save both server secrets securely."""

    api_key = getpass.getpass("Enter the server OpenAI API key (input is hidden): ").strip()
    if not api_key:
        raise ValueError("The OpenAI API key cannot be empty.")

    service_token = secrets.token_urlsafe(32)
    keyring = _keyring()
    keyring.set_password(
        SERVER_KEYRING_SERVICE_NAME,
        SERVER_OPENAI_KEY_NAME,
        api_key,
    )
    keyring.set_password(
        SERVER_KEYRING_SERVICE_NAME,
        SERVER_ACCESS_TOKEN_NAME,
        service_token,
    )

    print("\nLaptop server secrets were saved in Windows Credential Manager.")
    print("The OpenAI key will not be displayed or written to the repository.")
    print("\nCopy this access token to the approved NEAT client computers:")
    print(service_token)
    print("\nKeep this token private. It is not your OpenAI API key.")
    return service_token


def status() -> int:
    """Report whether both required credentials are present without revealing them."""

    has_key = bool(load_server_credential(SERVER_OPENAI_KEY_NAME))
    has_token = bool(load_server_credential(SERVER_ACCESS_TOKEN_NAME))
    print(f"OpenAI key saved: {'yes' if has_key else 'no'}")
    print(f"Shared access token saved: {'yes' if has_token else 'no'}")
    return 0 if has_key and has_token else 1


def remove() -> None:
    """Remove laptop server secrets from the OS credential manager."""

    keyring = _keyring()
    for name in (SERVER_OPENAI_KEY_NAME, SERVER_ACCESS_TOKEN_NAME):
        try:
            keyring.delete_password(SERVER_KEYRING_SERVICE_NAME, name)
        except keyring.errors.PasswordDeleteError:
            pass
    print("Laptop server secrets were removed from Windows Credential Manager.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure the laptop-hosted NEAT assistant server.",
    )
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()

    if args.status:
        raise SystemExit(status())
    if args.remove:
        remove()
        return
    configure()


if __name__ == "__main__":
    main()
