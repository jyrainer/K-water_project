# K-water_project
![K-water_project Logo](https://cdn.aifactory.space/images/20231018120320_NhMR.jpg)

**2023 제3회 K-water AI 경진대회** 어종(魚種) 식별 및 분류 알고리즘 개발

## 대회 주제
낙동강 하굿둑 물고기 영상에서 어종을 식별하고 분류하는 AI 모델 개발

## 진행내용
**일정:**
- 참가접수 및 팀빌딩: 10월 18일(수) ~ 11월 8일(수) 17:00
- 대회 기간: 10월 30일(월) ~ 11월 13일(월) 17:00
- 검증 기간(입상후보자 대상): 11월 14일(화) ~ 11월 20일(월)
- 결과 발표: 11월 20일(월) 15:00
- 시상식(오프라인): 11월 하순경 @대전

## Member
| 이름       | 학년 | 전공          | 역할                          |
|------------|-----|---------------|------------------------------|
| 박재용    | 4    | 전자공학부 | 데이터셋 분석, 학습 코드 작성 |
| 김경연    | 3    | 산업공학부 | 인터페이스 개선, 테스트 분석 |

## 개발 환경
- Python: 3.9.0

## 대회 규칙 및 검증 관련 안내
**1. 데이터 및 모델**
- 외부데이터 사용 불가능
- 사전학습모델(pre-trained)은 라이센스에 문제가 없다면 활용 가능
- 앙상블 가능
- Data augmentation 및 추가 라벨링 가능
- Test셋은 학습에 활용 불가

**2. 팀 참가**
- 한 팀의 인원 제한은 최대 4명
- 팀이 수상하는 경우 팀 대표에게만 상금 지급
- 시상식에는 입상팀 중 최소 1명 필참

**3. 제출**
- 제출횟수 제한: 팀당 1일 5회
- 추론 시간 제한: 1시간 이내
- 재제출 시간 제한: 1시간

**4. 저작물 제출 및 검증**
- 입상 후보팀으로 선정되는 경우 아래 3개 저작물을 cs@aifactory.page로 일괄 제출
- 코드와 주석의 인코딩은 모두 UTF-8을 사용
- 작성 코드: *.ipynb
- 최종 1회 제출
- 학습용 소스와 추론용 소스를 별도의 파일로 분리하는 것을 권장
- 검증자료 제출 시 사용한 Python 버전, OS 버전 필수 기재 (권장 버전: Python 3.9)
- 특히, 특수 패키지를 사용하는 경우 반드시 Python 패키지 명시
- 제출 시 랜덤 시드를 고정하지 않을 경우, 결과가 일괄적으로 산출되지 않을 수 있으므로 반드시 고정하여 제출 필수. (랜덤 시드 미고정시 입상대상에서 제외될 수 있음.)
- 모델 가중치(weight) 파일 또는 저장된 모델:
  - 딥러닝 계열로 weight가 파일로 저장되는 경우 저장된 weight
  - 그 밖의 경우는 pickle/ joblib 등의 라이브러리를 이용해 dump한 모델
- 모델 설명서: *.docx (양식 보기)
- 최종 1회 제출
- 입상자가 제출한 코드는 공지된 검증 기간 내 구동 및 성능에 대한 재현성 검증이 되어야 합니다.
- 모든 코드는 오류 없이 실행되어야 함.
- 별도 필요한 라이브러리가 있을 경우 소스코드 내에 설치하는 코드가 있어야 함.
- 원활한 코드 구동 및 성능 재현성 검증을 위해 필요한 최소한의 주석 혹은 가이드가 제공되어야 함.
