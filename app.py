from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def show_video_section() -> None:
    st.subheader("Annotated Output Video")
    video_path = OUTPUTS_DIR / "annotated_output.mp4"

    if file_exists(video_path):
        st.video(str(video_path))
        with video_path.open("rb") as fh:
            st.download_button(
                "Download annotated video",
                data=fh,
                file_name="annotated_output.mp4",
                mime="video/mp4",
            )
    else:
        st.info(
            "No annotated video found yet. Run the pipeline first:\n\n"
            "`python main.py --video path/to/your/video.mp4`"
        )


def show_analytics_section() -> None:
    st.subheader("Analytics Artifacts")
    analytics_files = [
        ("Heatmap", OUTPUTS_DIR / "heatmap.png"),
        ("Count Over Time", OUTPUTS_DIR / "count_over_time.png"),
        ("Trajectory Overlay", OUTPUTS_DIR / "trajectories.png"),
    ]

    found_any = False
    for title, path in analytics_files:
        if file_exists(path):
            found_any = True
            st.markdown(f"**{title}**")
            st.image(str(path), use_container_width=True)
            with path.open("rb") as fh:
                st.download_button(
                    f"Download {path.name}",
                    data=fh,
                    file_name=path.name,
                    mime="image/png",
                    key=f"download_{path.name}",
                )

    if not found_any:
        st.info("No analytics images found in `outputs/` yet.")


def show_docs_section() -> None:
    st.subheader("Documentation")

    readme_path = ROOT / "README.md"
    report_path = ROOT / "technical_report.md"

    col1, col2 = st.columns(2)
    with col1:
        if file_exists(readme_path):
            with readme_path.open("rb") as fh:
                st.download_button(
                    "Download README",
                    data=fh,
                    file_name="README.md",
                    mime="text/markdown",
                )
    with col2:
        if file_exists(report_path):
            with report_path.open("rb") as fh:
                st.download_button(
                    "Download technical report",
                    data=fh,
                    file_name="technical_report.md",
                    mime="text/markdown",
                )


def main() -> None:
    st.set_page_config(page_title="Multi-Object Tracking Demo", layout="wide")
    st.title("Multi-Object Detection and Persistent ID Tracking")
    st.caption("Live demo page for assignment submission")

    st.markdown(
        """
This app showcases generated artifacts from the pipeline:
- annotated tracking video with persistent IDs
- movement analytics (heatmap, count-over-time, trajectories)
- submission documentation
"""
    )

    with st.expander("How to generate artifacts locally", expanded=False):
        st.code(
            "python main.py --video path/to/your/video.mp4\n"
            "# or\n"
            "python main.py --url \"https://public-video-url\" --max_duration 0",
            language="bash",
        )

    show_video_section()
    st.divider()
    show_analytics_section()
    st.divider()
    show_docs_section()


if __name__ == "__main__":
    main()
