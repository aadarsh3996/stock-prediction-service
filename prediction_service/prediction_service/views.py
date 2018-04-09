import json
import requests
from django.shortcuts import render
from django.http import HttpResponse
from prediction_service import logistics,forest,deep_learning
import requests
import json
from django.http import JsonResponse



def logistic_regression(request):

    symbol=request.GET.get('symbol')
    print(symbol)

    final_dict=logistics.predict_logistic_regression(symbol)

    print(final_dict)

    return HttpResponse(json.dumps(final_dict), content_type="application/json")


def random_forest(request):

    symbol=request.GET.get('symbol')
    print(symbol)

    final_dict= forest.predict_forest(symbol)

    return HttpResponse(json.dumps(final_dict), content_type="application/json")

def neural_nets(request):

    symbol = request.GET.get('symbol')
    print(symbol)

    final_dict = deep_learning.predict_deep(symbol)

    return HttpResponse(json.dumps(final_dict), content_type="application/json")
