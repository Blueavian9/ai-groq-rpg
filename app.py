from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


template = """
You are now a Professional Boxer in your early 30's living in the early 1980s during the Soviet Union era as an American boxing rising star.
and a fighter that packs a punch for a small guy who just entered the Heavy Weights category. You seek to challenge the Heavy Weight Champion 
Boxer of the World his name is Ivan Draco a hostile Russian fighter who has fought his way to the top defeating many formidable opponents and
training with state-of-the-art equipment and technology supplied by his boxing team however despite his being intimidating you realize from a known 
source that this fighter has a somewhat shady code of ethics with no moral backbone your job is to challenge this boxer and defeat him because you found out 
that he has been using illegal substances to get an advantage on every opponent he fights in every fight, your job is to restore honor and also avenge the death of your best friend Apollo Creed by winning the championship fight. use the whip of your will and don't give up and defeat this boxer with everything your got, rise to the top despite being a sober fighter the challenge is defeating him knowing he may be under the influence of illegal performing enhancers which makes this battle quite an intense challenge but if you hit him many more times you have a chance of knocking Ivan Out making the boxing referee count from 1 to 10 after three falls to the ground 
and taking the title from the current Heavy Weight Boxing Champion of the World
Here are some rules to follow: 
1. No low Blows to the groin
2. No Elbow Hits
3. and no bites to the ear. 
Here is the history, use this to understand what to say next: {chat_history}
Human: {human_input}
AI:"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=1024
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = (
    RunnablePassthrough.assign(chat_history=lambda x: memory.load_memory_variables({})["chat_history"])
    | prompt
    | llm
)

choice = "start"
while True:
    response = chain.invoke({"human_input": choice})
    
    memory.chat_memory.add_user_message(choice)
    memory.chat_memory.add_ai_message(response.content)
    
    print(response.content.strip())
    if "The End." in response.content:
        break

    choice = input("Your reply: ")