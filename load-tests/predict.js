import http from 'k6/http';
import { check, sleep } from 'k6';
export const options = {
  stages: [{ duration: '1m', target: 100 }, { duration: '2m', target: 1000 },
           { duration: '5m', target: 1000 }, { duration: '1m', target: 0 }],
  thresholds: { http_req_failed: ['rate<0.01'], http_req_duration: ['p(95)<500', 'p(99)<1000'],
                checks: ['rate>0.99'] },
};
export default function () {
  const response = http.post(`${__ENV.BASE_URL || 'http://localhost:8080'}/api/predict`,
    JSON.stringify({ text: 'I enjoy exploring new ideas and understanding how systems work. I often spend time reflecting before making decisions and like working independently on creative projects.' }),
    { headers: { 'Content-Type': 'application/json' }, timeout: '10s' });
  check(response, { 'prediction returned': r => r.status === 200 && Boolean(r.json('label')) });
  sleep(1);
}
