import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor

ENDPOINT = "ENDPOINT_NAME"

def lambda_handler(event, context):

    image = base64.b64decode(event["body"]["image_data"])

    predictor = Predictor(ENDPOINT)
    predictor.serializer = IdentitySerializer("image/png")

    inferences = predictor.predict(image)

    event["body"]["inferences"] = inferences.decode('utf-8')
    
    return {
        'statusCode': 200,
        'body': event["body"]
    }