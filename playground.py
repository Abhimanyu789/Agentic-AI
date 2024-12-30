from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
import openai
from dotenv import load_dotenv

load_dotenv() 
import phi
from phi.playground import Playground,serve_playground_app

phi.api=os.getenv('PHI_API_KEY')

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
                    instructions=['Use tables to display the data.'],
                    show_tool_calls=True,
                    markdown=True)


app=Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__ == '__main__':
    serve_playground_app("playground:app",reload=True)
