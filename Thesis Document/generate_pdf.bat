@echo off
cd "C:\Users\yakim\Documents\MEGA\03. Vida Académica\03. Mestrado Ciencias Computadores\Dissertacao\Thesis Document"
cd Front
xelatex -interaction=nonstopmode -file-line-error main
xelatex -interaction=nonstopmode -file-line-error main

:: Clean up auxiliary files
del /q *.toc *.aux *.out *.bbl *.blg *.lof *.lot *.nav *.snm *.vrb *.synctex.gz *.fdb_latexmk *.fls *.xdv *.acn *.brf *.bglo *.glsdefs *.ist *.glo

cd ..
xelatex -interaction=nonstopmode -file-line-error main  :: Generating the auxiliary files
xelatex -interaction=nonstopmode -file-line-error main  :: This is to use the auxiliary files in the generation

:: Clean up auxiliary files
del /q *.toc *.aux *.out *.bbl *.blg *.lof *.lot *.nav *.snm *.vrb *.synctex.gz *.fdb_latexmk *.fls *.xdv *.acn *.brf *.bglo *.glsdefs *.ist *.glo

:: Optional: Alternative cleanup command
:: for /f "delims=" %%f in ('dir /b /a-d ^| findstr /v "main.tex main.pdf *.bat"') do del "%%f"