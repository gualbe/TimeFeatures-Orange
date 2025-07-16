
from setuptools import setup, find_packages
from distutils.command.install_data import install_data
from distutils.core import Command
from os import path, walk
import os
import sys
import subprocess

# Carpeta raíz donde está este archivo setup.py
here = path.abspath(path.dirname(__file__))

# Variable global para ir añadiendo archivos de datos
DATA_FILES = []

# Ajusta si usas namespaces
NAMESPACE_PACKAGES = ["TimeFeatures"]

def include_documentation():
    """
    Verifica la carpeta doc/_build/htmlhelp/html.
    Si no existe y estamos construyendo un wheel (bdist_wheel),
    intenta compilar la documentación con sphinx-build.
    """
    # Directorio fuente de la documentación (donde está conf.py)
    doc_source_dir = path.join(here, "doc")
    # Directorio de salida esperado (documentación compilada en HTML)
    doc_build_dir = path.join(doc_source_dir, "_build", "htmlhelp")
    # Directorio donde se instalará la documentación en el paquete
    install_dir = "help/orange3-example"

    # Verifica si existe la carpeta de salida
    if 'bdist_wheel' in sys.argv and not path.exists(doc_build_dir):
        print(f"No se encontró la documentación en '{doc_build_dir}'. "
              "Intentando construir la documentación con sphinx-build...")
        try:
            subprocess.check_call(["sphinx-build", "-b", "html",
                                   doc_source_dir, doc_build_dir])
        except Exception as e:
            print(f"Error al construir la documentación: {e}")
            sys.exit(1)

        if not path.exists(doc_build_dir):
            print("No se pudo construir la documentación. Abortando.")
            sys.exit(1)
    elif not path.exists(doc_build_dir):
        print(f"Advertencia: No existe '{doc_build_dir}'. "
              "La documentación no se incluirá.")
        return

    # Recorremos el directorio con la documentación compilada y la añadimos a DATA_FILES
    doc_files = []
    for dirpath, dirs, files in walk(doc_build_dir):
        # Remplazamos la ruta local por la ruta de instalación
        target_path = dirpath.replace(doc_build_dir, install_dir)
        doc_files.append((target_path, [path.join(dirpath, f) for f in files]))

    DATA_FILES.extend(doc_files)

include_documentation()

# === Cargar README como descripción ===
with open('README.md', 'r', encoding='utf-8') as f:
    ABOUT = f.read()

# === Datos generales ===
NAME = "TimeFeatures"
VERSION = "1.0.20"

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Environment :: Plugins',
    'Programming Language :: Python',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: OS Independent',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
]


# === Ruta a widgets y datos ===
PACKAGES = find_packages(include=["timefeatures", "timefeatures.*"])
PACKAGE_DATA = {
    "timefeatures.widgets": ["icons/*.svg", "icons/*.png"]
}

# === Entry points ===
ENTRY_POINTS={
        'orange.widgets': (
            'Time-Features = timefeatures.widgets',
        ),
        "orange.canvas.help": (
            'html-index = timefeatures.widgets:WIDGET_HELP_PATH',
        )
    }

print(DATA_FILES)

# === setup() ===
setup(
    name=NAME,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    entry_points=ENTRY_POINTS,
    data_files=DATA_FILES,
    author="Alejandro Rivas García",
    author_email="alejandrorivasgarcia@gmail.com",
    description="Timefeatures add-on for Orange 3 data mining software.",
    long_description=ABOUT,
    long_description_content_type='text/markdown',
    url="https://github.com/alervgr/Orange-TimeFeatures",
    license="GPL3+",
    classifiers=CLASSIFIERS,
)
