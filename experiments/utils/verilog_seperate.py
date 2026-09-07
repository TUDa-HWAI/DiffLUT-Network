import os
import torch
from difflogic import LogicLayer, LutlogicLayer, GroupSum
from difflogic.difflut_net import DiffLUTNetLayer


# ============================================================
#  Map 2-input logic operation ID to Verilog expression
# ============================================================
def logic_op_to_verilog(a, b, op):
    """
    a, b: Verilog variable name strings, e.g. "in_vec[3]"
    op: 0..15, corresponding to DiffLogic's 16 binary boolean operations
    """
    if op == 0:
        return "1'b0"
    elif op == 1:   # AND
        return f"({a} & {b})"
    elif op == 2:   # not_implies: A & ~B
        return f"({a} & ~{b})"
    elif op == 3:   # A
        return a
    elif op == 4:   # not_implied_by: B & ~A
        return f"({b} & ~{a})"
    elif op == 5:   # B
        return b
    elif op == 6:   # XOR
        return f"({a} ^ {b})"
    elif op == 7:   # OR
        return f"({a} | {b})"
    elif op == 8:   # NOR
        return f"~({a} | {b})"
    elif op == 9:   # XNOR
        return f"~({a} ^ {b})"
    elif op == 10:  # not B
        return f"~{b}"
    elif op == 11:  # implied_by: ~B | A
        return f"(~{b} | {a})"
    elif op == 12:  # not A
        return f"~{a}"
    elif op == 13:  # implies: ~A | B
        return f"(~{a} | {b})"
    elif op == 14:  # NAND
        return f"~({a} & {b})"
    elif op == 15:  # ONE
        return "1'b1"
    else:
        raise ValueError(f"Invalid logic operator ID: {op}")


# ============================================================
#  Generate a logic layer (2->1 LogicLayer) Verilog module
# ============================================================
def verilog_layer_logic(layer, layer_id, input_width):
    """
    :param layer: LogicLayer
    :param layer_id: int, used for module naming
    :param input_width: input vector width of the layer (previous out_dim)
    :return: (verilog_code_str, module_name, out_dim)
    """
    a_idx, b_idx = layer.indices  # shape [out_dim]
    w = layer.weights.argmax(-1).tolist()  # [out_dim]
    out_dim = layer.out_dim

    mod_name = f"layer_{layer_id}"
    lines = []
    lines.append(f"(* keep_hierarchy = \"yes\" *) module {mod_name}(input [{input_width-1}:0] in_vec, output [{out_dim-1}:0] out_vec);")


    for i in range(out_dim):
        ai = int(a_idx[i].item())
        bi = int(b_idx[i].item())
        op = int(w[i])
        expr = logic_op_to_verilog(f"in_vec[{ai}]", f"in_vec[{bi}]", op)
        lines.append(f"    assign out_vec[{i}] = {expr};")

    lines.append("endmodule\n")
    return "\n".join(lines), mod_name, out_dim


# ============================================================
#  Generate a LUT 3-2-1 structure layer Verilog module
# ============================================================
def verilog_layer_lut321(layer, layer_id, input_width):
    """
    :param layer: LutlogicLayer, internally assumed to be 3-2-1 six-gate expansion
    :return: (verilog_code_str, module_name, out_dim)
    """
    # Assuming indices in LutlogicLayer are 6 rows: a1..a6
    a1, a2, a3, a4, a5, a6 = layer.indices   # each shape [out_dim]
    # weights: [out_dim, 6, 16], first argmax to get 6 gate ids
    w = layer.weights.argmax(-1).tolist()    # [out_dim, 6]
    out_dim = layer.out_dim

    mod_name = f"layer_{layer_id}"
    lines = []
    lines.append(f"(* keep_hierarchy = \"yes\" *) module {mod_name}(input [{input_width-1}:0] in_vec, output [{out_dim-1}:0] out_vec);")

    for i in range(out_dim):
        # -------- First layer (g1,g2,g3) --------
        g1 = logic_op_to_verilog(
            f"in_vec[{int(a1[i].item())}]",
            f"in_vec[{int(a2[i].item())}]",
            int(w[i][0])
        )
        g2 = logic_op_to_verilog(
            f"in_vec[{int(a3[i].item())}]",
            f"in_vec[{int(a4[i].item())}]",
            int(w[i][1])
        )
        g3 = logic_op_to_verilog(
            f"in_vec[{int(a5[i].item())}]",
            f"in_vec[{int(a6[i].item())}]",
            int(w[i][2])
        )
        lines.append(f"    wire g1_{i} = {g1};")
        lines.append(f"    wire g2_{i} = {g2};")
        lines.append(f"    wire g3_{i} = {g3};")

        # -------- Second layer (g4,g5) --------
        g4 = logic_op_to_verilog(f"g1_{i}", f"g2_{i}", int(w[i][3]))
        g5 = logic_op_to_verilog(f"g2_{i}", f"g3_{i}", int(w[i][4]))
        lines.append(f"    wire g4_{i} = {g4};")
        lines.append(f"    wire g5_{i} = {g5};")

        # -------- Third layer output (g6) --------
        g6 = logic_op_to_verilog(f"g4_{i}", f"g5_{i}", int(w[i][5]))
        lines.append(f"    assign out_vec[{i}] = {g6};\n")

    lines.append("endmodule\n")
    return "\n".join(lines), mod_name, out_dim


# ============================================================
#  Generate top module: connect all layers + GroupSum
# ============================================================
def verilog_top_module(layer_infos, input_dim, last_dim, group_k=None, with_reg=False):
    """
    :param layer_infos: [(mod_name, in_dim, out_dim), ...]
    :param input_dim:  Model input bit width (after Flatten)
    :param last_dim:   Output width of the last Logic/LUT layer
    :param group_k:    GroupSum's k (number of classes), if None, no GroupSum, directly output last_dim bits
    :param with_reg:   If True, inserts flip-flops (registers) between the last logic layer and GroupSum adder tree
    """
    import math
    if group_k is None:
        final_out_dim = last_dim
        val_width = 1
    else:
        assert last_dim % group_k == 0, "Last layer dim must be divisible by GroupSum.k"
        per_class = last_dim // group_k
        val_width = int(math.ceil(math.log2(per_class + 1)))
        final_out_dim = group_k * val_width

    lines = []
    lines.append(f"module logic_net(")
    lines.append(f"    input clk,")
    lines.append(f"    input [{input_dim-1}:0] inp,")
    lines.append(f"    output reg [{final_out_dim-1}:0] out")
    lines.append(f");")
    lines.append("")

    prev_sig = "inp"

    # Connect all Logic/LUT layers combinationally
    for i, (mod_name, in_dim, out_dim) in enumerate(layer_infos):
        cur_sig = f"mid_{i}"
        lines.append(f"    wire [{out_dim-1}:0] {cur_sig};")
        lines.append(f"    {mod_name} u_{mod_name}(.in_vec({prev_sig}), .out_vec({cur_sig}));")
        prev_sig = cur_sig

    # prev_sig is now the combinational output of the last logic layer (width last_dim)

    # Insert optional registers between the logic layer and GroupSum adder tree
    if with_reg:
        lines.append("")
        lines.append("    // --- Middle Pipeline Register (FF between Layer and Adder Tree) ---")
        lines.append(f"    reg [{last_dim-1}:0] last_layer_reg;")
        lines.append("    always @(posedge clk) begin")
        lines.append(f"        last_layer_reg <= {prev_sig};")
        lines.append("    end")
        adder_input = "last_layer_reg"
    else:
        adder_input = prev_sig

    if group_k is None:
        # No GroupSum, register output directly
        lines.append("")
        lines.append("    always @(posedge clk) begin")
        lines.append(f"        out <= {adder_input};")
        lines.append("    end")
    else:
        lines.append("")
        lines.append(f"    wire [{final_out_dim-1}:0] out_comb;")

        if with_reg:
            lines.append("    // --- Pipelined balanced adder tree for each class ---")
            lines.append("    // Register policy: keep one FF after the logic network, then add one FF after every 3 adder levels.")
            for i in range(group_k):
                lines.append(f"    // Class {i} adder tree")
                current_terms = []
                for j in range(per_class):
                    idx = i * per_class + j
                    term_name = f"sum_c{i}_l0_{j}"
                    if val_width > 1:
                        term_expr = f"{{{val_width-1}'b0, {adder_input}[{idx}]}}"
                    else:
                        term_expr = f"{adder_input}[{idx}]"
                    lines.append(f"    wire [{val_width-1}:0] {term_name} = {term_expr};")
                    current_terms.append(term_name)

                level = 0
                levels_since_reg = 0
                while len(current_terms) > 1:
                    level += 1
                    next_terms = []
                    for j in range((len(current_terms) + 1) // 2):
                        left = current_terms[2 * j]
                        right_idx = 2 * j + 1
                        term_name = f"sum_c{i}_l{level}_{j}"
                        if right_idx < len(current_terms):
                            right = current_terms[right_idx]
                            lines.append(f"    wire [{val_width-1}:0] {term_name} = {left} + {right};")
                        else:
                            lines.append(f"    wire [{val_width-1}:0] {term_name} = {left};")
                        next_terms.append(term_name)

                    levels_since_reg += 1
                    if levels_since_reg == 3 and len(next_terms) > 1:
                        reg_terms = []
                        lines.append(f"    // Pipeline FF after adder level {level} for class {i}")
                        for j, term_name in enumerate(next_terms):
                            reg_name = f"sum_c{i}_l{level}_r{j}"
                            lines.append(f"    reg [{val_width-1}:0] {reg_name};")
                            reg_terms.append(reg_name)
                        lines.append("    always @(posedge clk) begin")
                        for reg_name, term_name in zip(reg_terms, next_terms):
                            lines.append(f"        {reg_name} <= {term_name};")
                        lines.append("    end")
                        current_terms = reg_terms
                        levels_since_reg = 0
                    else:
                        current_terms = next_terms

                lines.append(f"    assign out_comb[{i} * {val_width} +: {val_width}] = {current_terms[0]};")
                lines.append("")
        else:
            lines.append("    // --- Flat Adder Expressions for each class ---")
            for i in range(group_k):
                terms = []
                for j in range(per_class):
                    idx = i * per_class + j
                    if val_width > 1:
                        terms.append(f"{{{val_width-1}'b0, {adder_input}[{idx}]}}")
                    else:
                        terms.append(f"{adder_input}[{idx}]")
                expr = " + ".join(terms)
                lines.append(f"    assign out_comb[{i} * {val_width} +: {val_width}] = {expr};")
                lines.append("")
        
        lines.append("    // --- Register Output ---")
        lines.append("    always @(posedge clk) begin")
        lines.append("        out <= out_comb;")
        lines.append("    end")

    lines.append("endmodule\n")
    return "\n".join(lines)


# ============================================================
#  Generate a DiffLUT-Net layer Verilog module
# ============================================================
def verilog_layer_difflut_net(layer, layer_id, input_width):
    # Resolve input connections: learnable mapping uses argmax, fixed uses indices
    if layer.connections == 'learnable':
        mapping_indices = layer.mapping.weights.argmax(dim=0).cpu()  # [out_dim * 6]
        out_dim = layer.out_dim
        a1 = mapping_indices[0:out_dim]
        a2 = mapping_indices[out_dim:2*out_dim]
        a3 = mapping_indices[2*out_dim:3*out_dim]
        a4 = mapping_indices[3*out_dim:4*out_dim]
        a5 = mapping_indices[4*out_dim:5*out_dim]
        a6 = mapping_indices[5*out_dim:6*out_dim]
    else:
        a1, a2, a3, a4, a5, a6 = layer.indices
        out_dim = layer.out_dim

    w_probs = torch.clamp(layer.weights, 0.0, 1.0)
    w_hard = (w_probs > 0.5).int().tolist()

    mod_name = f"layer_{layer_id}"
    lines = []
    lines.append(f"(* keep_hierarchy = \"yes\" *) module {mod_name}(input [{input_width-1}:0] in_vec, output [{out_dim-1}:0] out_vec);")

    for i in range(out_dim):
        i1, i2, i3, i4, i5, i6 = (
            int(a1[i].item()), int(a2[i].item()), int(a3[i].item()), 
            int(a4[i].item()), int(a5[i].item()), int(a6[i].item())
        )
        
        # w_i[63] maps to inputs all 1s (MSB). Reversed join places w[63] at leftmost character.
        lut_str = "".join(str(b) for b in reversed(w_hard[i]))
        lines.append(f"    assign out_vec[{i}] = (64'b{lut_str} >> {{in_vec[{i1}], in_vec[{i2}], in_vec[{i3}], in_vec[{i4}], in_vec[{i5}], in_vec[{i6}]}}) & 1'b1;")

    lines.append("endmodule\n")
    return "\n".join(lines), mod_name, out_dim




def describe_withreg_pipeline(last_dim, group_k):
    """Return pipeline register stage info for Verilogwithreg GroupSum export."""
    import math
    if group_k is None:
        return {
            "total_register_stages": 2,
            "logic_output_register_stages": 1,
            "adder_tree_register_stages": 0,
            "output_register_stages": 1,
            "adder_levels": 0,
            "segments": [],
        }

    per_class = last_dim // group_k
    adder_levels = int(math.ceil(math.log2(per_class))) if per_class > 1 else 0

    # Matches verilog_top_module: insert one register after every 3 adder levels,
    # but only if more adder levels remain after that register point.
    adder_tree_register_stages = sum(
        1 for level in range(3, adder_levels + 1, 3) if level < adder_levels
    )

    segments = []
    previous = 0
    for level in range(3, adder_levels + 1, 3):
        if level < adder_levels:
            segments.append(level - previous)
            previous = level
    if adder_levels > previous:
        segments.append(adder_levels - previous)

    return {
        "total_register_stages": 1 + adder_tree_register_stages + 1,
        "logic_output_register_stages": 1,
        "adder_tree_register_stages": adder_tree_register_stages,
        "output_register_stages": 1,
        "adder_levels": adder_levels,
        "segments": segments,
    }


def print_withreg_pipeline_summary(last_dim, group_k):
    info = describe_withreg_pipeline(last_dim, group_k)
    print(
        "[Verilogwithreg] Pipeline register stages: "
        f"{info['total_register_stages']} "
        f"(logic-output={info['logic_output_register_stages']}, "
        f"adder-tree={info['adder_tree_register_stages']}, "
        f"output={info['output_register_stages']}); "
        f"adder_levels={info['adder_levels']}; "
        f"adder_segments={info['segments']}"
    )

# ============================================================
#  Main export function: export each layer module + top logic_net (with GroupSum)
# ============================================================
def export_verilog_modular(model, save_path, with_reg=False):
    """
    Automatically parse torch.nn.Sequential:
      - Flatten
      - LogicLayer / LutlogicLayer
      - GroupSum

    Export:
      - Several layer_X modules (one for each layer)
      - A logic_net top module, connecting all layers,
        and adding GroupSum aggregation at the end
    """
    layer_infos = []    # [(mod_name, in_dim, out_dim)]
    modules = []        # Verilog source code for each module
    input_dim = None
    prev_dim = None
    layer_id = 0

    group_k = None  # GroupSum.k

    for m in model:
        # Ignore Flatten (we use bit vector uniformly at the top)
        if isinstance(m, torch.nn.Flatten):
            continue

        # ===== Logic Layer / LUT Layer =====
        if isinstance(m, (LogicLayer, LutlogicLayer)):
            if prev_dim is None:
                prev_dim = m.in_dim
                input_dim = m.in_dim

            if isinstance(m, LogicLayer):
                code, mod_name, out_dim = verilog_layer_logic(m, layer_id, prev_dim)
            else:
                code, mod_name, out_dim = verilog_layer_lut321(m, layer_id, prev_dim)

            modules.append(code)
            layer_infos.append((mod_name, prev_dim, out_dim))

            prev_dim = out_dim
            layer_id += 1

        # ===== GroupSum Layer =====
        elif isinstance(m, GroupSum):
            group_k = m.k
            break

        else:
            raise TypeError(f"Unsupported module in model for Verilog export: {type(m)}")

    # Generate top module (with GroupSum)
    top_code = verilog_top_module(layer_infos, input_dim, prev_dim, group_k=group_k, with_reg=with_reg)
    modules.append(top_code)

    # Write to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(modules))

    print(f"Verilog modular structure successfully exported -> {save_path} (with_reg={with_reg})")
    if with_reg:
        print_withreg_pipeline_summary(prev_dim, group_k)


# ============================================================
#  Export all DiffLUT-Net layers as modules in one file
# ============================================================
def export_difflut_net_verilog(model, save_path, with_reg=False):
    """
    Export each layer code as a separate module, but write them all into a single .v file.
    """
    layer_infos = []    
    modules = []
    input_dim = None
    prev_dim = None
    layer_id = 0

    group_k = None  

    for m in model:
        if isinstance(m, torch.nn.Flatten):
            continue

        if isinstance(m, DiffLUTNetLayer):
            if prev_dim is None:
                prev_dim = m.in_dim
                input_dim = m.in_dim

            code, mod_name, out_dim = verilog_layer_difflut_net(m, layer_id, prev_dim)

            modules.append(code)
            layer_infos.append((mod_name, prev_dim, out_dim))
            prev_dim = out_dim
            layer_id += 1

        # ===== GroupSum Layer =====
        elif isinstance(m, GroupSum):
            group_k = m.k
            break
        else:
            raise TypeError(f"Unsupported module in model for Verilog export: {type(m)}")

    # Generate top module
    top_code = verilog_top_module(layer_infos, input_dim, prev_dim, group_k=group_k, with_reg=with_reg)
    modules.append(top_code)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(modules))
    print(f"Exported DiffLUT-Net Verilog -> {save_path} (with_reg={with_reg})")
    if with_reg:
        print_withreg_pipeline_summary(prev_dim, group_k)
