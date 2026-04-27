def generate_tikz_mps(num_qubits, gates):
    # 1. Заголовок и стили
    tikz_code = [
        "\\documentclass[tikz, border=10pt]{standalone}",
        "\\usetikzlibrary{shapes.geometric, positioning}",
        "\\begin{document}",
        "\\begin{tikzpicture}[",
        "    thick,",
        "    gamma/.style={circle, draw=black, minimum size=8mm, fill=blue!10, inner sep=0pt},",
        "    lambda/.style={diamond, draw=black, minimum size=6mm, fill=red!10, inner sep=0pt},",
        "    gate1/.style={rectangle, draw=black, minimum size=8mm, fill=green!10, rounded corners=1mm},",
        "    gate2/.style={rectangle, draw=black, minimum width=8mm, minimum height=3.8cm, fill=orange!10, rounded corners=1mm},",
        "    wire/.style={draw, thick}",
        "]\n"
    ]

    last_nodes = {}

    # 2. Инициализация (t=0)
    tikz_code.append("% --- Начальное состояние (t=0) ---")
    for q in range(1, num_qubits + 1):
        y_pos = -3 * (q - 1)
        node_name = f"G{q}"
        tikz_code.append(f"\\node[gamma] ({node_name}) at (0, {y_pos}) {{$\\Gamma_{{{q}}}$}};")
        last_nodes[q] = node_name

        if q < num_qubits:
            lam_y = y_pos - 1.5
            tikz_code.append(f"\\node[lambda] (L{q}) at (0, {lam_y}) {{$\\lambda_{{{q}}}$}};")

    tikz_code.append("\n% Вертикальные связи")
    path_elements = []
    for q in range(1, num_qubits):
        path_elements.extend([f"(G{q})", f"(L{q})"])
    path_elements.append(f"(G{num_qubits})")
    tikz_code.append(f"\\draw[wire] {' -- '.join(path_elements)};\n")

    # 3. Добавление гейтов (t=1, 2, ...)
    t = 1
    x_step = 2
    for gate in gates:
        name = gate["name"]
        targets = gate["qubits"]
        x_pos = t * x_step

        tikz_code.append(f"% --- Гейт {name} на {targets} (t={t}) ---")

        if len(targets) == 1:
            q = targets[0]
            y_pos = -3 * (q - 1)
            node_name = f"U_{t}_{q}"
            tikz_code.append(f"\\node[gate1] ({node_name}) at ({x_pos}, {y_pos}) {{${name}$}};")
            
            # Соединяем с предыдущим узлом на этом проводе
            tikz_code.append(f"\\draw[wire] (0, {y_pos} -| {last_nodes[q]}.east) -- (0, {y_pos} -| {node_name}.west);")
            last_nodes[q] = node_name

        elif len(targets) == 2:
            q1, q2 = min(targets), max(targets)
            y_mid = -3 * (q1 - 1) - 1.5
            node_name = f"V_{t}_{q1}_{q2}"
            
            # Размещаем 2-кубитный гейт ровно посередине между проводами
            tikz_code.append(f"\\node[gate2] ({node_name}) at ({x_pos}, {y_mid}) {{${name}$}};")
            
            # Проводим две линии к левой границе большого гейта
            y_pos_1 = -3 * (q1 - 1)
            y_pos_2 = -3 * (q2 - 1)
            tikz_code.append(f"\\draw[wire] (0, {y_pos_1} -| {last_nodes[q1]}.east) -- (0, {y_pos_1} -| {node_name}.west);")
            tikz_code.append(f"\\draw[wire] (0, {y_pos_2} -| {last_nodes[q2]}.east) -- (0, {y_pos_2} -| {node_name}.west);")
            
            last_nodes[q1] = node_name
            last_nodes[q2] = node_name

        t += 1
        tikz_code.append("")

    # 4. Финальные открытые индексы (выходы)
    tikz_code.append("% --- Открытые индексы (выходы) ---")
    x_out = t * x_step
    for q in range(1, num_qubits + 1):
        out_node = f"out{q}"
        y_pos = -3 * (q - 1)
        tikz_code.append(f"\\coordinate ({out_node}) at ({x_out}, {y_pos});")
        
        # Тянем линию от восточного края последнего узла к выходу
        tikz_code.append(f"\\draw[wire] (0, {y_pos} -| {last_nodes[q]}.east) -- ({out_node});")
        tikz_code.append(f"\\node[right] at ({out_node}) {{$|i_{{{q}}}\\rangle$}};")

    tikz_code.append("\\end{tikzpicture}")
    tikz_code.append("\\end{document}")

    return "\n".join(tikz_code)

# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ:
# ==========================================
if __name__ == "__main__":
    # Задаем список гейтов (обрати внимание, кубиты теперь с 1)
    test_gates = [
        {"name": "V", "qubits": [1, 2]},
    ]
    
    # Генерируем код для 3 кубитов
    latex_output = generate_tikz_mps(num_qubits=2, gates=test_gates)
    
    # Сохраняем в файл
    with open("mps_circuit.tex", "w", encoding="utf-8") as f:
        f.write(latex_output)
        
    print("Файл mps_circuit.tex успешно сгенерирован!")