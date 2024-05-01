from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
    {
        "question": """음주운전 2회로 집행유예 기간 중, 무면허 음주운전으로 적발되었습니다. 저는 업무상 차량이 필수적인 직업을 가지고 있어, 
        면허 취소 이후에도 집행유예 기간동안 타인의 명의로 몰래 차량을 이용해왔습니다. 최근 다시 음주운전을 하게 되었고, 그러던 중 차 안에서 잠이 들어 경찰에 적발되었습니다. 
        저는 술을 마시고 약 7시간 뒤에 적발되었고, 측정 결과 혈중알코올농도는 0.083이었습니다. 
        부모님께도 죄송하고, 특히 자력으로는 생활할 수 없는 가족을 돌봐야 하는 상황에서 실형을 피할 수 있는 방법과 면허를 구제받을 수 있는 방법이 있는지 알고 싶습니다.""",
        "answer": """
안녕하세요, 판례 데이터 분석 기반 상담 챗봇입니다.
먼저, 집행유예 기간 중에 범죄를 저질러 금고 이상의 형을 선고받아 확정되면 집행유예가 실효되고 그 유예된 형까지 같이 복역해야 합니다. 따라서, 이번 사건에서는 벌금형을 선고받아야 실형을 피할 수 있습니다.
그러나, 혈중알코올농도가 0.08% 이상인 경우에는 1년 이상 2년 이하의 징역 또는 500만원 이상 1천만원 이하의 벌금에 처해질 수 있으므로, 질문자님의 경우에는 실형을 피하기 어려울 가능성이 높습니다.

또한, 면허 취소에 대해서는 이의신청이나 행정심판을 청구하여 다툴 수 있고, 기각되면 행정소송을 제기하여 구제받을 수 있으나, 이 또한 쉽지 않을 것으로 보입니다.

따라서, 질문자님께서는 변호사와 상담하여 적극적으로 대응하시는 것이 좋습니다. 변호사는 질문자님의 상황을 분석하여 적절한 대응 전략을 제시하고, 재판에서 유리한 판결을 이끌어내는 데 도움을 줄 수 있습니다.

저는 기존 판례에 기반한 정보를 제공하고 있습니다. 사안마다 구체적인 상황이 다르므로 정확한 법률적 판단을 위해서는 변호사와 상담하시는 것을 추천드립니다.
""",
    },
    {
        "question": """최근 전애인으로부터 저를 죽여버리겠다는 협박 전화를 받았습니다. 이에 저는 협박죄로 고소하기 위해 준비를 하면서 전애인에게 내용증명을 보냈습니다. 
        그런데 상대방은 저로 인해 자신이 받은 모욕이 더 크고, 제가 보낸 내용증명이 자신을 협박한 것이라고 주장하며 소송을 제기하겠다고 연락을 해오고 있습니다. 
        저는 상대방을 모욕한 적은 일절 없으며, 아직 고소는 진행하지 않았습니다.
1. 가해자가 저에게 계속 연락을 해오는데, 상습협박죄에 해당하는지 알고 싶습니다.
2. 제가 상대방에게 고소할 예정이라고 내용증명을 보낸 것이 협박죄에 해당하는지 궁금합니다.
3. 상대방에게 오는 연락에 답장을 하고 싶은데, 공격적으로 답장을 하면 협박죄로 고소할 때 불리하게 작용할지 걱정됩니다. 답장을 하지 않고 무시하는 것이 나을까요?""",
        "answer": """
안녕하세요, 판례 데이터 분석 기반 상담 챗봇입니다.

1. 협박죄에서 협박이란 일반적으로 보아 사람으로 하여금 공포심을 일으킬 정도의 해악을 고지하는 것을 의미합니다. 
이러한 협박 행위가 여러 차례 반복된다면 상습협박죄가 성립할 수 있습니다. 
하지만, 단순히 연락을 자주 하는 것만으로는 상습협박죄가 성립한다고 보기 어렵습니다.
2. 고소 예정이라는 내용증명서를 보낸 것만으로는 협박죄에 해당하지 않습니다. 
단, 내용증명서에 상대방에게 공포심을 유발할 수 있는 내용이 포함되어 있다면 협박죄가 성립할 수 있으니 유의해야 합니다.

3. 상대방의 행동 또는 발언으로 인해 위협을 느끼신다면 경찰에 신고하거나 전문가와 상담하시어 안전을 확보하시길 바랍니다. 
또, 감정적으로 흥분된 상태에서 상대방에게 답장을 할 경우 불필요한 갈등이 발생하거나 자신에게 불리한 증거가 남을 수 있으므로, 시간을 두고 신중하게 생각한 후에 답장을 보내는 것이 좋습니다.
저는 기존 판례에 기반한 정보를 제공하고 있습니다. 
사안마다 구체적인 상황이 다르므로 정확한 법률적 판단을 위해서는 변호사와 상담하시는 것을 추천드립니다.
""",
    },
    {
        "question":""" 어머니께서 이혼 절차를 밟고 계셨습니다. 그런데 법원에 마지막으로 출석만 하면 이혼이 완료되는 상황에서 돌아가셨습니다. 
        이 경우 이혼 처리는 어떻게 되는지 알고 싶습니다. 이혼 성립이 되는 것인가요?
1. 어머니께서 돌아가셨으므로, 사망신고 여부와 관계없이 당사자가 사망했기 때문에 이혼이 성립되지 않는 것인가요?
2. 아니면 법원 출석일 전날까지 사망신고를 완료하고, 이혼 출석을 사망신고서로 제출하면 이혼 처리가 가능한건가요? """,
        "answer": """
안녕하세요, 판례 데이터 분석 기반 상담 챗봇입니다.

이혼소송은 혼인관계를 해소하는 것이 목적인데, 당사자가 사망한 경우에는 이혼소송의 목적이 이미 달성되었기 때문에 소송을 계속할 필요가 없게 됩니다. 따라서 이혼소송은 종료되고, 법원은 이혼 여부에 대한 판단은 하지 않고 막바로 소송이 종결되었다는 의미에서 '소송종료선언'이라는 결정을 하게 됩니다.

따라서 법원에 사망신고서 등을 제출하는 방법으로 당사자의 사망 사실을 알려야 하고, 만약 법원이 사망사실을 모르고 이혼 판결을 하더라도 그 판결은 아무런 효력이 없습니다.

결과적으로 이혼 전에 당사자가 사망한 것이 되므로 상속인들은 이혼을 고려함이 없이 일반적인 사망과 마찬가지로 상속관계 등을 처리하면 됩니다.

저는 판례에 기반한 정보를 제공하고 있습니다. 사안마다 구체적인 상황이 다르므로 정확한 법률적 판단을 위해서는 변호사와 상담하시는 것을 추천드립니다.
""",
    },
    
]