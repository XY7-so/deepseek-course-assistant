import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 读取本地 .env 文件
load_dotenv()


# 页面设置
st.set_page_config(
    page_title="DeepSeek课程资料助手",
    page_icon="🤖",
    layout="wide"
)


# 页面标题
st.title("DeepSeek课程资料智能助手")
st.write("上传课程资料后，可以针对资料内容向 DeepSeek 提问。")


# 侧边栏系统信息
with st.sidebar:
    st.header("系统信息")
    st.write("基础模型：DeepSeek")
    st.write("应用框架：Streamlit")
    st.write("检索方法：TF-IDF")
    st.write("微调方法：LoRA")
    st.write("应用场景：课程资料智能问答")

    st.divider()

    st.write("使用步骤：")
    st.write("1. 上传课程资料")
    st.write("2. 输入课程问题")
    st.write("3. 查看 DeepSeek 回答")
    st.write("4. 查看参考资料片段")


# 读取 API Key
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    api_key = os.getenv("DEEPSEEK_API_KEY")


# 检查 API Key
if not api_key:
    st.error(
        "没有检测到 DeepSeek API Key。"
        "本地运行时请配置 .env 文件，"
        "部署到 Streamlit 时请配置 Secrets。"
    )
    st.stop()


# 创建 DeepSeek 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


def read_uploaded_file(uploaded_file):
    """
    读取 TXT、PDF、DOCX 文件内容。
    """
    file_name = uploaded_file.name.lower()

    # 读取 TXT
    if file_name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    # 读取 PDF
    if file_name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    # 读取 DOCX
    if file_name.endswith(".docx"):
        document = Document(uploaded_file)

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

    return ""


def split_text(text, chunk_size=800, overlap=100):
    """
    将长文本切分成多个片段。
    """
    chunks = []
    start = 0

    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += step

    return chunks


def retrieve_chunks(question, chunks, top_k=3):
    """
    使用 TF-IDF 检索与问题最相关的文本片段。
    """
    if not chunks:
        return []

    if len(chunks) <= top_k:
        return chunks

    try:
        vectorizer = TfidfVectorizer()

        matrix = vectorizer.fit_transform(
            chunks + [question]
        )

        scores = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        )[0]

        indexes = scores.argsort()[-top_k:][::-1]

        return [chunks[i] for i in indexes]

    except Exception:
        return chunks[:top_k]


# 上传课程资料
uploaded_file = st.file_uploader(
    "上传课程资料",
    type=["txt", "pdf", "docx"],
    help="支持 TXT、PDF 和 DOCX 文件"
)


course_text = ""
document_chunks = []


# 读取上传的文件
if uploaded_file is not None:
    course_text = read_uploaded_file(uploaded_file)

    if course_text.strip():
        st.success(f"已成功读取：{uploaded_file.name}")
        st.info(f"资料长度：{len(course_text)} 个字符")

        document_chunks = split_text(course_text)

        st.info(
            f"资料已切分为 {len(document_chunks)} 个文本片段"
        )
    else:
        st.warning(
            "文件中没有读取到有效文字。"
            "如果是扫描版 PDF，可能需要先进行 OCR 文字识别。"
        )


# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []


# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 输入问题
question = st.chat_input("请输入课程相关问题")


if question:
    # 保存用户问题
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # 显示用户问题
    with st.chat_message("user"):
        st.markdown(question)


    # 构造 Prompt
    if document_chunks:
        relevant_chunks = retrieve_chunks(
            question,
            document_chunks,
            top_k=3
        )

        context = "\n\n".join(relevant_chunks)

        # 显示检索到的资料片段
        with st.expander("查看本次检索到的资料片段"):
            for index, chunk in enumerate(
                relevant_chunks,
                start=1
            ):
                st.markdown(f"**资料片段 {index}**")
                st.write(chunk)

        prompt = f"""
请根据以下检索到的课程资料片段回答问题。

课程资料片段：
{context}

用户问题：
{question}

回答要求：
1. 优先使用课程资料片段中的信息；
2. 如果资料中没有足够信息，请明确说明“资料中没有找到足够信息”；
3. 不要编造课程资料中不存在的内容；
4. 使用清晰、准确、适合大学生理解的中文；
5. 先给出结论，再进行解释；
6. 回答最后列出使用到的参考资料片段。
"""

    else:
        st.info(
            "当前未上传课程资料，回答将直接由 DeepSeek 生成。"
        )

        prompt = f"""
请回答下面的问题：

{question}

回答要求：
1. 使用清晰、准确、适合大学生理解的中文；
2. 先给出结论，再进行解释；
3. 对不确定的内容进行说明；
4. 不要编造没有依据的信息。
"""


    # 调用 DeepSeek
    with st.chat_message("assistant"):
        with st.spinner("DeepSeek 正在生成回答……"):

            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": """
你是一名严谨的人工智能课程学习助手。

请遵守以下规则：
1. 优先根据用户上传的课程资料回答问题；
2. 如果资料中没有足够信息，请明确说明“资料中没有找到足够信息”；
3. 不要编造课程资料中不存在的内容；
4. 使用清晰、准确、适合大学生理解的中文；
5. 回答时先给出结论，再进行解释；
6. 如果使用了课程资料，请在回答最后列出参考资料片段；
7. 对不确定的内容要说明不确定性。
"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1200
                )

                answer = response.choices[0].message.content

                if not answer:
                    answer = "模型没有返回有效回答，请重新提问。"

                st.markdown(answer)

                # 保存模型回答
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:
                st.error(
                    "调用 DeepSeek 失败，请检查 API Key、网络连接和模型名称。"
                )

                with st.expander("查看错误详情"):
                    st.code(str(error))


# 清空对话按钮
if st.button("清空对话"):
    st.session_state.messages = []
    st.rerun()