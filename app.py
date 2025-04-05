import streamlit as st
import asyncio
import nest_asyncio
from typing import Dict, Any
from langchain_core.output_parsers import PydanticOutputParser
from langchain_wrappers import (
    PoemAnalysisChain,
    AudioGenerationChain,
    ImagePromptChain,
    VideoCompilationChain
)
from langchain_wrappers.llm_wrapper import GeminiLLM
from langchain_wrappers.prompts import NARRATOR_INSTRUCTION, PROMPTER_INSTRUCTION
from langchain_wrappers.schema_models import Visualization, Narration
import pandas as pd

# Allow nested event loops in Streamlit
nest_asyncio.apply()


# Function to run image and audio generation in parallel
async def image_and_audio_parallel(input_data: Dict[str, Any]) -> Dict[str, Any]:
    image_prompt_chain = ImagePromptChain(
        llm=GeminiLLM(system_instruction=PROMPTER_INSTRUCTION),
        output_parser=PydanticOutputParser(pydantic_object=Visualization)
    )
    audio_generation_chain = AudioGenerationChain()

    image_result, audio_result = await asyncio.gather(
        image_prompt_chain.arun({"narration_segments": input_data["narration"]["segments"]}),
        audio_generation_chain.arun({"audio_texts": input_data["narration"]})
    )
    return {"image_prompts": image_result, "audios": audio_result}


def main():
    st.title("Poem Analysis & Multimedia Generation")
    st.write("Paste your poem text and poet's name below to generate analysis, images, and audio.")

    # Session state to track analysis progress
    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False
    if "video_output" not in st.session_state:
        st.session_state.video_output = None

    # Show input fields only if analysis hasn't started
    if not st.session_state.analyze_clicked:
        poem_name = st.text_input("📜 Poem Name", placeholder="Enter title of the poem")
        poet_name = st.text_input("✍️ Poet Name", placeholder="Enter the poet's name")
        poem_body = st.text_area("📖 Poem Body", height=300, placeholder="Paste the full poem text here")

    # Analyze button
    if st.button("🚀 Analyze Poem") and not st.session_state.analyze_clicked:
        if not poet_name or not poem_body:
            st.error("⚠️ Please provide both the poet's name and the poem text.")
            return
        else:
            st.session_state.analyze_clicked = True
            progress_bar = st.progress(0)

            # Separate status placeholders
            status_poem = st.empty()
            status_multimedia = st.empty()
            status_video = st.empty()

            # Step 1: Poem Analysis
            with status_poem.container():
                with st.spinner("🔍 Analyzing Poem..."):
                    poem_chain = PoemAnalysisChain(
                        llm=GeminiLLM(system_instruction=NARRATOR_INSTRUCTION),
                        output_parser=PydanticOutputParser(pydantic_object=Narration)
                    )
                    poem_analysis_result = poem_chain.invoke({"poem_text": poem_body})
            status_poem.success("✅ Poem Analysis Completed")
            progress_bar.progress(33)

            # Step 2: Display Poem Analysis
            segments = poem_analysis_result.get('narration', {}).get('segments', [])
            if not isinstance(segments, list):
                segments = []

            df = pd.DataFrame(segments).astype(str)
            df = df.applymap(lambda x: x.strip())
            df = df[~(df == "").all(axis=1)]
            df['segment'] = df['segment'].astype(str)
            df.set_index('segment', inplace=True)

            st.subheader("📊 Poem Analysis Result")
            st.data_editor(df, height=300, use_container_width=True, hide_index=False, num_rows="fixed")

            # Step 3: Multimedia Generation
            with status_multimedia.container():
                with st.spinner("🎨 Generating Multimedia (Images & Audio)..."):
                    async def run_chains():
                        return await image_and_audio_parallel(poem_analysis_result)
                    loop = asyncio.get_event_loop()
                    results = loop.run_until_complete(run_chains())
            status_multimedia.success("✅ Multimedia Generation Completed")
            progress_bar.progress(66)

            # Step 4: Video Compilation
            with status_video.container():
                with st.spinner("🎥 Compiling Video..."):
                    final = {
                        "poem_name": poem_name,
                        "poet": poet_name,
                        "poem_lines": [seg["lines"] for seg in segments],
                        "images": results["image_prompts"],
                        "audios": results["audios"]
                    }
                    video_compiler = VideoCompilationChain()
                    compiler_output = video_compiler.run(final)
            status_video.success("✅ Video Compilation Completed!")
            progress_bar.progress(100)

            st.session_state.video_output = compiler_output

    # Final Output
    if st.session_state.video_output:
        st.subheader("🎬 Generated Video Output")
        st.video(st.session_state.video_output)
        st.download_button(
            label="📥 Download Video",
            data=st.session_state.video_output,
            file_name="poem_video.mp4",
            mime="video/mp4"
        )


if __name__ == "__main__":
    main()
