from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

information = """
    James Clerk Maxwell FRSE FRS (13 June 1831 – 5 November 1879) was a Scottish mathematician[1][2] and scientist responsible for the classical theory of electromagnetic radiation, which was the first theory to describe electricity, magnetism and light as different manifestations of the same phenomenon. Maxwell's equations for electromagnetism have been called the "second great unification in physics"[3] where the first one had been realised by Isaac Newton.

    With the publication of "A Dynamical Theory of the Electromagnetic Field" in 1865, Maxwell demonstrated that electric and magnetic fields travel through space as waves moving at the speed of light. He proposed that light is an undulation in the same medium that is the cause of electric and magnetic phenomena.[4] The unification of light and electrical phenomena led to his prediction of the existence of radio waves. Maxwell is also regarded as a founder of the modern field of electrical engineering.[5]

    Maxwell helped develop the Maxwell–Boltzmann distribution, a statistical means of describing aspects of the kinetic theory of gases. He is also known for presenting the first durable colour photograph in 1861 and for his foundational work on analysing the rigidity of rod-and-joint frameworks (trusses) like those in many bridges.

    His discoveries helped usher in the era of modern physics, laying the foundation for such fields as special relativity and quantum mechanics. Many physicists regard Maxwell as the 19th-century scientist having the greatest influence on 20th-century physics. His contributions to the science are considered by many to be of the same magnitude as those of Isaac Newton and Albert Einstein.[6] In the millennium poll—a survey of the 100 most prominent physicists—Maxwell was voted the third greatest physicist of all time, behind only Newton and Einstein.[7] On the centenary of Maxwell's birthday, Einstein described Maxwell's work as the "most profound and the most fruitful that physics has experienced since the time of Newton".[8] Einstein, when he visited the University of Cambridge in 1922, was told by his host that he had done great things because he stood on Newton's shoulders; Einstein replied: "No I don't. I stand on the shoulders of Maxwell."
"""

if __name__ == "__main__":
    print("Hello LangChain!")

    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=information))
