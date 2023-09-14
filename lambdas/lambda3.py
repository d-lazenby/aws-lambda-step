import json

THRESHOLD = 0.95

def lambda_handler(event, context):

    # Grab the inferences from the event
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = (True if max(eval(event['body']['inferences'])) > THRESHOLD else False)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    try:
        assert meets_threshold
    except:
        raise AssertionError("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event['body']
    }
