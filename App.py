from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="openai/gpt-oss-20b")

STYLE_PROMPT = """
You are an experienced blog writer.

Write in a natural, human-like, clear, and engaging style.

Avoid generic AI phrases such as:
"furthermore", "unlock", "leverage", "robust",
"seamless", "game-changer", "it's worth noting",
"navigate the landscape", "at the end of the day".

Do not start every section in the same way.
Use bullet points only when the content genuinely needs a list.
Prefer meaningful paragraphs.
Avoid unnecessary repetition.
Do not end with a summary of what you just said.
End with one useful final thought.
"""


# 2. Outline Prompt
OUTLINE_PROMPT = ChatPromptTemplate.from_template(
    "Write a short outline for a blog post.\n\n"
    "Topic: {topic}\n"
    "Tone: {tone}\n\n"
    "Give me:\n"
    "- One opening line that makes the reader curious\n"
    "- Four section headings\n"
    "- Five key points the post must cover\n\n"
    "Just the outline. Do not write the post."
)


# 3. Draft Prompt
DRAFT_PROMPT = ChatPromptTemplate.from_template(
    "Write a blog post from this outline.\n\n"
    "Tone: {tone}\n\n"
    "Outline:\n{outline}\n\n"
    "Instructions:\n"
    "- Start with the opening line from the outline\n"
    "- Develop all four section headings\n"
    "- Cover all five key points\n"
    "- Write clear, engaging paragraphs\n"
    "- Explain difficult concepts in simple language\n"
    "- Avoid repetition and generic AI phrases\n"
    "- Follow the requested tone\n"
    "- End with one useful final thought\n"
)

POLISH_PROMPT = ChatPromptTemplate.from_template(
    "Polish the following blog post to make it clear, engaging, "
    "and natural to read.\n\n"

    "Topic: {topic}\n"
    "Tone: {tone}\n\n"

    "Blog Draft:\n{draft}\n\n"

    "Instructions:\n"
    "- Improve sentence structure and readability\n"
    "- Make the content flow naturally between sections\n"
    "- Replace awkward phrases with clear, natural language\n"
    "- Remove unnecessary repetition and filler words\n"
    "- Keep the headings meaningful and engaging\n"
    "- Preserve the original meaning and important information\n"
    "- Maintain the requested tone\n"
    "- Follow these writing guidelines:\n{style}\n\n"

    "Return only the polished blog post. "
    "Do not include explanations about the changes."
)

outline_chain = OUTLINE_PROMPT | llm | StrOutputParser()
draft_chain = DRAFT_PROMPT | llm | StrOutputParser()
polish_prompt = POLISH_PROMPT | llm | StrOutputParser() 

st.subheader("AI Blog Generator")
st.caption("Generate a Block using three prompt chain")
topic = st.text_input(
    "Enter your blog topic",
    placeholder="How to learn GenAI in 2026"
)
tone = st.selectbox("Tone",["direct","normal","friendly"])
if st.button("Generate",type="primary"):
    with st.spinner("Generating the outline"):
        outline = outline_chain.invoke({"topic":topic,"tone":tone})

    with st.spinner("Generating the Draft"):
        draft = draft_chain.invoke({"outline":outline,"tone":tone})

    with st.spinner("polishing the prompt"):
        final = polish_prompt.invoke({"topic":topic,"tone":tone,"draft":draft,"style":STYLE_PROMPT})


    st.markdown(final)

    with st.expander("raw Blog",expanded=False):
        st.code(final)






