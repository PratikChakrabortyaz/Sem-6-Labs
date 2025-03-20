from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings


def magazine_cover_form(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']  # Get the uploaded image
        background_color = request.POST.get('background_color')
        text = request.POST.get('text')
        font_size = request.POST.get('font_size')
        font_color = request.POST.get('font_color')

        # Store the image in the media folder
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(image.name, image)
        image_url = fs.url(filename)  # Get the URL of the uploaded image

        print(f"Image URL: {image_url}")
        # Pass all form data to the context, including the image URL
        context = {
            'image_url': image_url,
            'background_color': background_color,
            'text': text,
            'font_size': font_size,
            'font_color': font_color,
        }

        # Render the preview page with context data
        return render(request, 'magazine/preview_cover.html', context)

    return render(request, 'magazine/magazine_cover_form.html')

def preview_cover(request):
    return render(request, 'magazine/preview_cover.html')