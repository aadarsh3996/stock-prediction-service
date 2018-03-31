import json
import requests
from django.shortcuts import render
from django.http import HttpResponse
from prediction_service import logistics
import requests
import json
from django.http import JsonResponse



def logistic_regression(request):

    symbol=request.GET.get('symbol')
    print(symbol)

    final_dict=logistics.predict_logistic_regression(symbol)

    print(final_dict)

    return HttpResponse(json.dumps(final_dict), content_type="application/json")
