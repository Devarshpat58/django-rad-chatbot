# Static Images Directory

This directory contains static images for the Django RAG Chatbot.

## Directory Structure

- **icons/**: Application icons and small graphics
- **backgrounds/**: Background images and patterns
- **logos/**: Company and application logos
- **illustrations/**: Decorative illustrations and graphics

## Image Guidelines

### Formats
- **SVG**: Preferred for icons and simple graphics
- **PNG**: For images with transparency
- **JPG**: For photographs and complex images
- **WebP**: Modern format for better compression (when supported)

### Naming Convention
- Use lowercase letters
- Use hyphens to separate words
- Be descriptive: `property-search-icon.svg`
- Add suffix for variants: `logo-dark.svg`, `logo-light.svg`

### Optimization
- Compress images before adding to the project
- Use appropriate dimensions (avoid oversized images)
- Consider providing multiple resolutions for high-DPI displays

## Usage in Templates

```django
{% load static %}
<img src="{% static 'images/icons/search.svg' %}" alt="Search">
```

## Current Images

*Add descriptions of images as they are added to the project*
