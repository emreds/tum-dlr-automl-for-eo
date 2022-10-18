
## Steps to build documentation

1. Build and install package from source.
```bash
cd ./tum-dlr-automl-for-eo/
pip install -e .
pip install sphinx sphinx_rtd_theme
```

2. Generate documentation from your docstrings.
```bash
cd docs/
sphinx-apidoc -f -o ./source ../src/tum_dlr_automl_for_eo
```
3. Build the documentation
```bash
make clean && make html
```
4. You can now view your documentation under `docs/build/html/index.html`.
