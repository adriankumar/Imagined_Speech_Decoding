Spherical_Harmonics/
  __init__.py
  __main__.py        #python -m sh_sandbox
  app.py             #pywebview window
  web/
    index.html
    style.css
    vendor/          #three.min.js + OrbitControls.js (you download these, below)
    render.js        #three.js scene scaffolding
    main.js          #boot



pip install pywebview
To run, ensure you are outside this directory and enter python -m sh_sandbox