pip3 uninstall pytom3d -y
python3 -m build
pip3 install dist/pytom3d-0.0.0rc2-py3-none-any.whl
sphinx-apidoc -o ./docs/ ./src/
cd docs/
make clean & make html
cd ..
