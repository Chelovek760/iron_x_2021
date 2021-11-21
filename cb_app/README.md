Докер

> docker run --rm -it -v $(pwd):/app -p 7860:7860 streamlit:test bash

Запуска
> streamlit run app.py --server.port 7860 --server.address 0.0.0.0