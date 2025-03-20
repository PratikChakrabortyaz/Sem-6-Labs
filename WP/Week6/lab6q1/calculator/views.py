# calculator/views.py
from django.shortcuts import render

def calculator_view(request):
    result = None
    if request.method == 'POST':
        try:
            num1 = int(request.POST.get('num1'))
            num2 = int(request.POST.get('num2'))
            operation = request.POST.get('operation')

            # Perform the selected operation
            if operation == 'add':
                result = num1 + num2
            elif operation == 'subtract':
                result = num1 - num2
            elif operation == 'multiply':
                result = num1 * num2
            elif operation == 'divide':
                if num2 != 0:
                    result = num1 / num2
                else:
                    result = 'Error: Division by zero'
        except ValueError:
            result = 'Error: Invalid input'
    
    return render(request, 'calculator/calculator.html', {'result': result})
