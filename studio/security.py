from __future__ import annotations

import secrets

from flask import flash, jsonify, redirect, request, session


def init_csrf(app) -> None:
    @app.context_processor
    def inject_csrf_token():
        return {"csrf_token": get_csrf_token}

    @app.before_request
    def protect_against_csrf():
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None

        sent_token = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
        expected_token = session.get("_csrf_token")
        if expected_token and secrets.compare_digest(sent_token or "", expected_token):
            return None

        message = "La requête n’a pas pu être vérifiée. Actualisez la page puis réessayez."
        if request.is_json or request.path.startswith("/api/"):
            return jsonify({"error": message}), 400

        flash(message, "error")
        return redirect(request.url)


def get_csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(24)
        session["_csrf_token"] = token
    return token
