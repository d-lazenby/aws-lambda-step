import json

THRESHOLD = 0.95


def lambda_handler(event, context):
    meets_threshold = (True if max(eval(event['body']['inferences'])) > THRESHOLD else False)

    try:
        assert meets_threshold
    except:
        raise AssertionError("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event['body']
    }
