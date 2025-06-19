from enum import Enum


class MCQ_Type(Enum):
    LETTER = "letter"
    YES_OR_NO = "yes_or_no"


DEFAULT_MCQ_PROMPT = {
    "default": """Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.

Provide the letter of the best option wrapped in parentheses within the <answer> </answer> tags.

Question: {}
{}""",
    "r1": """Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question. 

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags. 

Then, provide the letter of the best option wrapped in parentheses within the <answer> </answer> tags.

Question: {}
{}""",
    "r1_nocot": """Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question. 

Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags. 

Then, provide the letter of the best option wrapped in parentheses within the <answer> </answer> tags.

Question: {}
{}<think>
</think>""",
}

DEFAULT_TG_PROMPT = {
    "default": """To accurately pinpoint the event "{}" in the video, determine the precise time period of the event.

Provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".""",
    "r1": """To accurately pinpoint the event "{}" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".""",
    "r1_nocot": """To accurately pinpoint the event "{}" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83".
<think>
</think>
""",
}
