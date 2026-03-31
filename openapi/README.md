# OpenAPI Specification for s2.cpp HTTP Server

This directory contains the OpenAPI 3.1 specification for the s2.cpp HTTP server mode.

## File

- `s2-openapi.yaml` – OpenAPI 3.1 YAML specification

## Overview

The s2.cpp binary includes an HTTP server mode (`--server`) that exposes generation and persistent voice-cloning endpoints.

### Endpoint

- `POST /generate` – Accepts multipart/form-data with text, optional reference audio, and generation parameters; returns a WAV audio file.
- `POST /clone` – Accepts multipart/form-data with a voice title, reference audio, and transcript; returns a saved voice id plus metadata.

### Server Defaults

- Host: `127.0.0.1`
- Port: `3030`

## Usage with OpenAPI Tools

You can use tools like [Swagger UI](https://swagger.io/tools/swagger-ui/), [Redoc](https://redoc.ly/), or [OpenAPI Generator](https://openapi-generator.tech/) to render documentation, generate client code, or validate requests.

Example with `curl`:

```bash
curl -X POST http://127.0.0.1:3030/generate \
  -F "text=Hello world" \
  -F "params={\"temperature\":0.9}" \
  -o output.wav

curl -X POST http://127.0.0.1:3030/clone \
  -F "title=Alice" \
  -F "file=@reference.wav" \
  -F "transcript=Transcript of the reference audio"
```

## Notes

- The server implements `/generate` and `/clone`; no health‑check or root endpoint is provided.
- Reference audio must be accompanied by its transcript (`reference_text` field).
- Saved voice ids returned by `/clone` can be supplied back to `/generate` through the `voice` or `voice_id` multipart field.
- The submitted `title` becomes the `voice_id`, and saving with the same title overwrites the existing `title.s2voice` profile.
- The `params` field is a JSON string that follows the `GenerateParams` schema defined in the OpenAPI document.

## License

Same as the s2.cpp project (MIT).