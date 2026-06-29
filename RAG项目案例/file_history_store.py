import os,json
from typing import Sequence


from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import message_to_dict,messages_from_dict,BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory


def get_history(session_id):
    return FileChatMessageHistory(session_id,"chat_history")








class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id        #会话id
        self.storage_path = storage_path    #不同会话ID的存储文件，所在的文件夹路径
        #完整的文件路径
        self.file_path = os.path.join(self.storage_path,self.session_id)
        #确保文件夹确实存在
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    def add_messages(self,messages: Sequence[BaseMessage])  -> None:
        all_messages = list(self.messages)
        all_messages.extend(messages)

        new_messages = []
        for message in all_messages:
            d = message_to_dict(message)
            new_messages.append(d)

        #new_messages = [message_to_dict(message) for message in all_messages]
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f)

    @property  #装饰器将messages方法变成成员属性用
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messges_data = json.load(f)
                return messages_from_dict(messges_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)
