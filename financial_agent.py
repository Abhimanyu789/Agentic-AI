from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import os
import openai
from dotenv import load_dotenv
load_dotenv() 
openai.api_key = os.getenv('OPENAI_API_KEY')

#Web Search Agent
web_search_agent=Agent(name='Web Search Agent', 
                       role ='Search the web for information', 
                       model=Groq(id='llama3-groq-70b-8192-tool-use-preview'), 
                       tools=[DuckDuckGo()],
                       isntructions='Always include the source of the information in your response.',
                       show_tool_calls=True,
                       markdown=True)


## Financial Agent
finance_agent=Agent(name='Financial Agent',
                    role='Provide financial information',
                    model=Groq(id='llama3-groq-70b-8192-tool-use-preview'),
                    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)],
                    instructions='Use tables to display the data.',
                    show_tool_calls=True,
                    markdown=True)

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    instructions=["Always include the source of the information in your response.","Use tables to display the data."],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("Summarize Financial Analyst recommendation and share the latest news for Apple Inc. (AAPL).",stream=True)