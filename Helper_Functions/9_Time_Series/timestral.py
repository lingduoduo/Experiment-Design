"""Run the Timestral example without loading a model on import.

Reusable helpers live in timestral_example.py to avoid a name collision with the
installed timestral package. Run this file with --help for options.
"""

if __name__ == "__main__":
    from timestral_example import main

    raise SystemExit(main())
