"""Entry point for Vercel: it looks for a Flask object named `app`."""
from webapp import create_app

app = create_app()
