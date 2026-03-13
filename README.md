# 랜덤 벽 GridWorld에서 TD Learning 구현

강화학습 TD Learning 아이디어를 **직접 코드로 실험**해보는 작은 프로젝트입니다.  
4×4 GridWorld 환경과, **매 에피소드마다 벽이 랜덤으로 생성되는 GridWorld**를 만들고,  
**랜덤 정책**으로 움직이는 에이전트가 **TD(0)** 로 상태가치 함수 \(V(s)\)를 학습하도록 구현했습니다.

이 저장소는 블로그 글:

- **“랜덤 벽 GridWorld에서 TD Learning으로 상태가치 함수 배우기”**

를 뒷받침하는 실험 코드/노트북입니다.

---

## 1. 핵심 아이디어

- 4×4 GridWorld에서:
  - 시작 상태: `(0, 0)`
  - 목표 상태: `(3, 3)`
  - 행동: `0=왼쪽`, `1=위`, `2=오른쪽`, `3=아래`
- 보상:
  - 일반 이동: `-1`
  - 목표 도착: 에피소드 종료 (보상 0으로 처리)
- TD(0) 업데이트:
  \[
  V(s) \leftarrow V(s) + \alpha \bigl(r + \gamma V(s') - V(s)\bigr)
  \]

두 가지 환경을 비교합니다.

1. **기본 GridWorld** – 벽이 전혀 없는 4×4 그리드  
2. **랜덤 벽 GridWorld** – 매 에피소드마다 2~3개의 벽을 랜덤으로 생성하는 그리드

두 환경 모두 **정책은 완전 랜덤**(각 방향 25%),  
차이는 **환경 구조(벽 유무)** 가 상태가치 \(V(s)\)에 어떻게 반영되는지입니다.

---

## 2. 환경 설명

### 2.1 기본 4×4 GridWorld

- 4×4 그리드, 좌표 `(0,0)` ~ `(3,3)`
- 시작: `(0, 0)`
- 목표: `(3, 3)`
- 행동:
  - `0`: 왼쪽
  - `1`: 위
  - `2`: 오른쪽
  - `3`: 아래
- 이동 규칙:
  - 그리드 밖으로 나가면 **제자리 유지**
  - 나머지는 한 칸씩 이동
- 보상:
  - 이동: `-1`
  - 목표 도착: `done=True`, 보상 0

### 2.2 랜덤 벽 GridWorld

- 위 기본 환경과 동일한 그리드/시작/목표/보상 구조
- 추가 규칙 – **벽(Walls)**:
  - 매 에피소드 시작 시 `_generate_walls()`로 벽 생성
  - 벽 개수: 2개 또는 3개
  - 벽 위치 제약:
    - 시작 `(0,0)`에는 벽을 두지 않음
    - 목표 `(3,3)`에도 벽을 두지 않음
    - 같은 위치에 벽 중복 생성 X
  - 이동 규칙:
    - 이동하려는 칸이 벽이면 **제자리 유지**
    - 그리드 밖으로 나가도 **제자리 유지**

이렇게 하면 에이전트 입장에서는:

> “기본 지도”에서는 항상 같은 길로 갈 수 있지만,  
> “랜덤 벽 지도”에서는 매 에피소드마다  
> 길이 열릴 수도, 막힐 수도 있는 환경을 만나는 셈입니다.

---

## 3. 구현 개요

노트북: `랜덤_벽_GridWord에서_TD_Learning_구현.ipynb`

주요 구성:

### 3.1 공통 설정

```python
N = 4
START = (0, 0)
GOAL = (3, 3)
ACTIONS = [0, 1, 2, 3]  # 0: left, 1: up, 2: right, 3: down

GAMMA = 0.9
ALPHA = 0.1
EPISODES = 1000
```

- 작은 환경이기 때문에 `EPISODES=1000` 정도로도 충분히 패턴이 나옵니다.

### 3.2 기본 4×4 GridWorld 환경 클래스

```python
class BasicGridWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent_pos = START
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos

        if action == 0:      # left
            y = max(0, y - 1)
        elif action == 1:    # up
            x = max(0, x - 1)
        elif action == 2:    # right
            y = min(N - 1, y + 1)
        elif action == 3:    # down
            x = min(N - 1, x + 1)

        self.agent_pos = (x, y)

        if self.agent_pos == GOAL:
            return self.agent_pos, 0.0, True
        else:
            return self.agent_pos, -1.0, False
```

### 3.3 랜덤 벽 GridWorld 환경 클래스

```python
class RandomWallGridWorld:
    def __init__(self):
        self.walls = set()
        self.reset()

    def _generate_walls(self):
        self.walls.clear()
        num_walls = random.choice([2, 3])
        candidates = [(i, j) for i in range(N) for j in range(N)
                      if (i, j) not in (START, GOAL)]
        random.shuffle(candidates)
        for pos in candidates:
            if len(self.walls) >= num_walls:
                break
            self.walls.add(pos)

    def reset(self):
        self._generate_walls()
        self.agent_pos = START
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        nx, ny = x, y

        if action == 0:      # left
            ny = max(0, y - 1)
        elif action == 1:    # up
            nx = max(0, x - 1)
        elif action == 2:    # right
            ny = min(N - 1, y + 1)
        elif action == 3:    # down
            nx = min(N - 1, x + 1)

        if (nx, ny) in self.walls:
            nx, ny = x, y

        self.agent_pos = (nx, ny)

        if self.agent_pos == GOAL:
            return self.agent_pos, 0.0, True
        else:
            return self.agent_pos, -1.0, False
```

### 3.4 TD(0) 학습 루프 (랜덤 정책)

```python
from collections import defaultdict

def td_learn(env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
             max_steps=100, name=""):
    V = defaultdict(float)
    for ep in range(episodes):
        s = env.reset()
        done = False

        for t in range(max_steps):
            if done:
                break

            # 완전 랜덤 정책
            a = random.choice(ACTIONS)
            s_prime, r, done = env.step(a)

            old_v = V[s]
            target = r + (gamma * V[s_prime] if not done else r)
            V[s] = old_v + alpha * (target - old_v)

            s = s_prime

        if (ep + 1) % 100 == 0 and name:
            print(f\"[{name}] episode {ep+1}/{episodes}\")
    return V
```

- `max_steps`를 두어, 벽 배치 때문에 목표에 못 가는 경우에도 **무한 루프에 빠지지 않도록** 제한합니다.
- 정책은 완전 랜덤이지만, TD가 **기대 Return을 근사**하면서 V(s)를 업데이트합니다.

### 3.5 상태가치 테이블 출력

```python
def print_value_table(V, title):
    print(title)
    for i in range(N):
        row = []
        for j in range(N):
            v = V[(i, j)]
            row.append(f\"{v:6.2f}\")
        print(\" ".join(row))
    print()
```

---

## 4. 실행 방법

### 4.1 환경

- Python 3.x
- 별도 외부 라이브러리는 없고, 표준 라이브러리(`random`, `collections`)만 사용

### 4.2 실행

노트북 안에서:

```python
if __name__ == "__main__":
    random.seed(2026)

    # 기본 GridWorld 학습
    basic_env = BasicGridWorld()
    V_basic = td_learn(basic_env, name="basic")
    print_value_table(V_basic, "[기본 4x4 GridWorld 상태가치 V(s)]")

    # 랜덤 벽 GridWorld 학습
    wall_env = RandomWallGridWorld()
    V_wall  = td_learn(wall_env, name="wall")
    print_value_table(V_wall, "[랜덤 벽 GridWorld 상태가치 V(s)]")
```

실행하면, 에피소드 진행 로그와 함께 두 환경의 V(s) 테이블이 출력됩니다.

---

## 5. 예시 출력 및 해석

예를 들어, 다음과 같은 결과가 나올 수 있습니다.

```text
[basic] episode 100/1000
...
[basic] episode 1000/1000
[기본 4x4 GridWorld 상태가치 V(s)]
 -9.33  -9.12  -8.79  -8.46
 -9.20  -8.91  -8.56  -7.35
 -8.81  -8.38  -7.10  -4.28
 -8.62  -7.64  -4.59   0.00

[wall] episode 100/1000
...
[wall] episode 1000/1000
[랜덤 벽 GridWorld 상태가치 V(s)]
 -9.51  -9.41  -9.09  -8.81
 -9.39  -9.21  -8.59  -7.40
 -9.19  -9.01  -7.34  -4.66
 -9.07  -8.55  -5.23   0.00
```

- (3,3)은 목표 상태이므로 두 환경 모두 V = 0
- **기본 4×4**:
  - 목표에서 멀어질수록(위, 왼쪽으로 갈수록) 값이 점점 더 큰 음수 →  
    목표까지 평균적으로 더 많은 `-1` 보상을 받는 위치
- **랜덤 벽 4×4**:
  - 같은 칸이라도 기본 환경보다 **더 큰 음수(더 나쁜 값)** 인 경우가 많음
  - 예:
    - (0,0): 기본 -9.33 vs 벽 -9.51  
    - (2,2): 기본 -7.10 vs 벽 -7.34  

이는:

> 랜덤 벽 환경에서는,  
> 같은 위치에서도 벽 때문에 **자주 막히거나 돌아가야 해서**  
> 평균적으로 더 많은 `-1`을 맞게 되고 → V(s)가 더 낮아진다

는 뜻입니다.

---

## 6. 한 줄 정리

**같은 4×4 GridWorld라도,  
벽이 랜덤으로 생기는 환경에서는 “자주 막히는 위험한 칸”의 상태가치가  
기본 환경보다 훨씬 더 나쁜 값으로 학습된다.**

TD Learning은 단순 랜덤 정책만 가지고도  
환경 구조(벽, 막힘)를 상태가치 함수 \(V(s)\) 안에 자연스럽게 녹여낸다는 점을  
이 작은 예제로 확인할 수 있습니다.

