FROM lychee-python-sandbox:local

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --root-user-action=ignore --no-cache-dir -i "${PIP_INDEX_URL}" \
    beautifulsoup4 \
    lxml \
    numpy \
    openpyxl \
    pandas \
    pdfplumber \
    pillow \
    pydub \
    pypdf \
    python-docx \
    requests \
    xlrd
