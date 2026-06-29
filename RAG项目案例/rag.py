from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, format_document,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda

from file_history_store import get_history
from vector_stores import VectorStoreService
import config_data as config


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagService(object):

    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model =config.embedding_model_name )
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的参考资料为主。"
                 "简洁和专业的回答用户问题。参考资料：{context}"),
                ("system","并且我提供用户的对话历史记录，如下"),
                MessagesPlaceholder("history"),
                ("user","请回答用户提问：{input}"),

            ]
        )

        self.chat_model = ChatTongyi(model = config.chat_model_name, streaming=True)

        self.chain = self.__get_chain()


    def __get_chain(self):
        """获取最终执行链"""
        retriever = self.vector_service.get_retriever()
        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段: {doc.page_content}\n文档原数据: {doc.metadata}\n\n"

            return formatted_str

        def temp1(value: dict) -> str:
            return value["input"]

        def temp2(value):
            new_value = {}
            new_value["input"] = value["input"]
            new_value["context"] = value["context"]
            # 如果没有 history 键，默认返回空列表
            new_value["history"] = value.get("history", [])
            return new_value




        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(temp1) | retriever | format_document
            } | RunnableLambda(temp2) |self.prompt_template | print_prompt |self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )



        return conversation_chain


if __name__ == "__main__":
    #session id 配置
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
    res = RagService().chain.invoke({"input":"春天穿什么衣服"},session_config)
    print(res)