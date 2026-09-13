# Deploy NCEV AO ASM1 Simulator to Streamlit Community Cloud

## Goal
After deployment, users only open a browser link. No Python installation is required on their computers.

## Files that must be in the GitHub repository
- app.py
- asm1_engine.py
- fractionation.py
- requirements.txt
- .streamlit/config.toml

The other files are documentation/testing only.

## Step 1 — Create a GitHub repository
Create a new repository, for example:

`ncev-asm1-simulator`

For company/proprietary use, prefer a **private repository**.

## Step 2 — Upload this package
Upload the contents of this folder to the repository root.

The repository root should look like:

```text
ncev-asm1-simulator/
├── app.py
├── asm1_engine.py
├── fractionation.py
├── requirements.txt
├── smoke_test.py
├── README.md
└── .streamlit/
    └── config.toml
```

## Step 3 — Deploy on Streamlit Community Cloud
1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app.
3. Select the GitHub repository.
4. Main file path: `app.py`
5. Choose Python 3.12 or 3.13 if the deployment UI asks for a Python version.
6. Deploy.

## Step 4 — Share the link
You will receive a URL similar to:

`https://your-app-name.streamlit.app`

Send that URL to users. They need only Chrome/Edge/Safari.

## Updating the app
Push updated code to GitHub. The hosted app can rebuild/reload from the repository.

## Troubleshooting
If deployment fails:
- Check the Streamlit build log.
- Confirm `requirements.txt` is at repository root.
- Confirm the main file is `app.py`.
- Prefer Python 3.12/3.13 if a newer runtime causes SciPy dependency issues.

## Security
This MVP does not provide user accounts, project permissions, or a database.
Do not enter sensitive customer information into a public/shared app until access control and data policy are defined.
