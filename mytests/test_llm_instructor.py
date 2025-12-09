import instructor
from litellm import completion
from pydantic import BaseModel, field_validator,Field

# Integration with litellm
client = instructor.from_litellm(completion, mode=instructor.Mode.MD_JSON)


# class User(BaseModel):
#     name: str
#     age: int


class User(BaseModel):
    name: str = Field(description="User's name")
    age: int = Field(description="User's age")

    @field_validator('age')
    def validate_age(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v


# Create structured output
# user = client.chat.completions.create(
#     model="dashscope/qwen-plus",
#     messages=[
#         {"role": "user", "content": "Extract: Jason is 25 years old"},
#     ],
#     response_model=User,
#     max_retries=3
# )

# print('\n*****************')
# print(user)  # User(name='Jason', age=25)

# loop and receive user input and print output
while True:
    user_input = input("Enter a message: ")
    user = client.chat.completions.create(
        model="dashscope/qwen-plus",
        messages=[
            {"role": "user", "content": user_input},
        ],
        response_model=User,
    )
    print("\n*****************")
    print(user)
