from dandelion import DataTXT
import pprint
datatxt = DataTXT(token=["79e9105159624c008882b568e933580f"])
response = datatxt.sim('Barack Obama is the president of the US',
             'Bob Iger is the CEO of Walt Disney')




pprint.pprint(response)



response = datatxt.sim('Barack Obama is the president of the US',
             'Barack Obama is elected the president of the US')




pprint.pprint(response)