# Frontend React

Frontend nay la man hinh interactive cho ESG scoring, gui `txt` len backend va hien log realtime.

## Chay local

Backend API:

```bash
python api_server.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Mac dinh:

- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`

## Luong du lieu

1. Nguoi dung nhap `company`, `sector`, `year`, `report_text`
2. Frontend `POST /api/score`
3. Backend ghi input thanh file `.txt` tam va goi `run_pipeline(...)`
4. Frontend nghe `GET /api/jobs/:id/stream` de cap nhat log
5. Khi job xong, frontend tai ket qua JSON tu `GET /api/jobs/:id`
