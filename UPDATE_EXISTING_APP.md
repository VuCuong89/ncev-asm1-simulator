# Update your existing Streamlit app from Cloud v1 to Cloud v2.1

Your current GitHub repository already contains the working v1 app.
The safest update is to **keep v1 as rollback** and upload v2 as a new folder.

## Recommended method

1. In GitHub repository `ncev-asm1-simulator`, click **Add file → Upload files**.
2. Upload the complete folder:

   `ncev_asm1_cloud_deploy_v2`

3. Commit changes.
4. Open Streamlit Community Cloud.
5. Open your current app → **Settings**.
6. Change the main file path from the v1 path to:

   `ncev_asm1_cloud_deploy_v2/app.py`

7. Save / reboot the app.
8. Wait for the build to finish and test the default case.

## Rollback

If v2 has a problem, change the main file path back to the existing v1 `app.py` path.

This method avoids deleting or overwriting the working version.
