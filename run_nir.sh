#!/bin/bash
# Запуск скрипта НИР в tmux (сессия не умрёт при дисконнекте)
# Использование: ./run_nir.sh 01_data_prep.py
#                ./run_nir.sh attach       — подключиться к существующей сессии

SESSION="nir"
SCRIPT="${1:-}"

if [ "$SCRIPT" = "attach" ] || [ -z "$SCRIPT" ]; then
    tmux attach -t "$SESSION" 2>/dev/null || echo "Нет активной сессии '$SESSION'"
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Сессия '$SESSION' уже существует. Подключаюсь..."
    tmux attach -t "$SESSION"
else
    cd /Users/jam_malina/Desktop/nir2
    tmux new-session -d -s "$SESSION" -x 220 -y 50
    tmux send-keys -t "$SESSION" "cd /Users/jam_malina/Desktop/nir2 && python3 -u $SCRIPT 2>&1 | tee logs/$(basename $SCRIPT .py)_\$(date +%Y%m%d_%H%M%S).log" Enter
    echo "Запущен '$SCRIPT' в tmux-сессии '$SESSION'"
    echo "Подключиться: tmux attach -t $SESSION"
    echo "Отключиться (не убивая):  Ctrl+B, затем D"
    tmux attach -t "$SESSION"
fi
