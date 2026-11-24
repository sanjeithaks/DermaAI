
from pathlib import Path

app_file = Path("app.py")

if not app_file.exists():
    raise SystemExit("ERROR: app.py not found. Open this script in your DermaAI project folder.")

source = app_file.read_text(encoding="utf-8")

if "Capture & Analyse from Raspberry Pi" in source:
    print("Camera integration is already present.")
    raise SystemExit(0)

# Add HTTP imports
if "import urllib.request" not in source:
    source = source.replace(
        "import json",
        "import json\nimport urllib.request\nimport urllib.error",
        1
    )

# Add Raspberry Pi address
base_line = "BASE_DIR = Path(__file__).resolve().parent"

if "PI_CAMERA_URL =" not in source:
    if base_line not in source:
        raise SystemExit("ERROR: BASE_DIR not found. No changes made.")

    source = source.replace(
        base_line,
        base_line + '\nPI_CAMERA_URL = "http://10.34.66.20:8081"',
        1
    )

# Find the existing upload section
anchor = "    uploaded_file = st.file_uploader("

if anchor not in source:
    raise SystemExit("ERROR: Image uploader not found. No changes made.")

camera_code = '''
    st.markdown("### 📷 Raspberry Pi camera")

    if st.button("📷 Capture & Analyse from Raspberry Pi",
                 use_container_width=True):

        try:
            with st.spinner("Capturing image and checking skin..."):

                request = urllib.request.Request(
                    f"{PI_CAMERA_URL}/capture-analyze",
                    data=b"",
                    method="POST"
                )

                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))

                if not payload.get("ok"):
                    raise RuntimeError("Pi capture failed.")

                capture = payload.get("capture", {})
                gate = payload.get("analysis", {})
                quality = gate.get("quality", {})
                status = gate.get("status", "")

                if status != "skin_detected" or not quality.get("ok", False):

                    st.session_state["analysis_result"] = None
                    st.session_state["analysis_image"] = None

                    st.warning(
                        "Cancer prediction skipped. The Pi did not approve "
                        "this image as good-quality skin."
                    )

                    st.write("Pi result:", status)

                    reasons = quality.get("reasons", [])
                    if reasons:
                        st.write("Image-quality issues:", ", ".join(reasons))

                else:

                    image_url = PI_CAMERA_URL + capture["url"]

                    image_request = urllib.request.Request(image_url)

                    with urllib.request.urlopen(
                        image_request, timeout=60
                    ) as response:
                        image_bytes = response.read()

                    captured_image = Image.open(
                        BytesIO(image_bytes)
                    ).convert("RGB")

                    with st.spinner(
                        "Skin detected. Running cancer prediction..."
                    ):
                        result = predict_image(captured_image)

                    save_history(capture["file"], result)

                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_image"] = captured_image.copy()

                    st.success("Camera capture and analysis completed!")

                    st.caption(
                        "Pi skin-detection score: "
                        + str(gate.get("skin", {}).get("p_skin", "N/A"))
                        + ". This is not a cancer probability."
                    )

        except urllib.error.URLError as error:
            st.error(
                "Cannot connect to Raspberry Pi. Check that both devices "
                "are connected to the same network. " + str(error)
            )

        except Exception as error:
            st.error("Camera analysis failed: " + str(error))

'''

# Insert the camera feature before the existing uploader
source = source.replace(anchor, camera_code + "\n" + anchor, 1)

# Check Python syntax before saving
compile(source, "app.py", "exec")

# Save a safety backup if one does not already exist
backup = Path("app.py.camera_backup")

if not backup.exists():
    backup.write_text(app_file.read_text(encoding="utf-8"), encoding="utf-8")

app_file.write_text(source, encoding="utf-8")

print("SUCCESS: Raspberry Pi camera button added to app.py.")
print("Backup:", backup.name)
print("Pi address:", "http://10.34.66.20:8081")
