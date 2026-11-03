`timescale 1ns/1ps
// Minimal RV32I-like single-cycle core (subset) with simple data mem interface
//
// Opravy 2026-11-03:
//  - pametove rozhrani je kombinacni (jednocyklove jadro): adresa/data/strobe
//    plati ve stejnem taktu jako instrukce a LW zapisuje data z teto adresy.
//    Drive se adresa registrovala a LW ulozil data z predchozi instrukce.
//  - zapis do x0 je ignorovan (drive se pri rd=0 x0 prepsal).
//  - vlastni AI instrukce ma format R (opcode 1111011, funct3=000):
//      rs1 = base1, rs2 = base2, funct7 = delka (0..127).
//    Drive I-format, kde pole rs2 kolidovalo s bity okamzite hodnoty.
module core(
    input  wire         clk,
    input  wire         rst,
    input  wire [31:0]  instr,    // instruction from IMEM
    output reg  [31:0]  pc,

    // simple data memory interface (kombinacni; mem_rdata musi byt asynchronni)
    output reg  [31:0]  mem_addr,
    output reg  [31:0]  mem_wdata,
    output reg  [3:0]   mem_wstrb,
    output reg          mem_write,
    output reg          mem_read,
    input  wire [31:0]  mem_rdata,

    // accelerator interface (custom instruction, jednotaktovy puls)
    output reg          accel_start,
    output reg  [31:0]  accel_base1,
    output reg  [31:0]  accel_base2,
    output reg  [31:0]  accel_len
);

    reg [31:0] regs [0:31];
    wire [6:0] opcode = instr[6:0];
    wire [2:0] funct3 = instr[14:12];
    wire [6:0] funct7 = instr[31:25];
    wire [4:0] rd = instr[11:7];
    wire [4:0] rs1 = instr[19:15];
    wire [4:0] rs2 = instr[24:20];

    function [31:0] imm_i(input [31:0] i);
        imm_i = {{20{i[31]}}, i[31:20]};
    endfunction
    function [31:0] imm_s(input [31:0] i);
        imm_s = {{20{i[31]}}, i[31:25], i[11:7]};
    endfunction
    function [31:0] imm_b(input [31:0] i);
        imm_b = {{19{i[31]}}, i[31], i[7], i[30:25], i[11:8], 1'b0};
    endfunction
    function [31:0] imm_j(input [31:0] i);
        imm_j = {{11{i[31]}}, i[31], i[19:12], i[20], i[30:21], 1'b0};
    endfunction

    wire [31:0] rs1_val = (rs1 == 5'd0) ? 32'd0 : regs[rs1];
    wire [31:0] rs2_val = (rs2 == 5'd0) ? 32'd0 : regs[rs2];

    // dekodovani: zapis do registru a dalsi PC (kombinacne)
    reg        wb_en;
    reg [31:0] wb_val;
    reg [31:0] pc_next;

    always @* begin
        mem_addr = 32'd0;
        mem_wdata = 32'd0;
        mem_wstrb = 4'd0;
        mem_write = 1'b0;
        mem_read = 1'b0;
        wb_en = 1'b0;
        wb_val = 32'd0;
        pc_next = pc + 32'd4;
        case (opcode)
            7'b0110011: begin // R-type
                wb_en = 1'b1;
                case ({funct7, funct3})
                    {7'b0000000, 3'b000}: wb_val = rs1_val + rs2_val; // ADD
                    {7'b0100000, 3'b000}: wb_val = rs1_val - rs2_val; // SUB
                    {7'b0000000, 3'b111}: wb_val = rs1_val & rs2_val; // AND
                    {7'b0000000, 3'b110}: wb_val = rs1_val | rs2_val; // OR
                    {7'b0000000, 3'b100}: wb_val = rs1_val ^ rs2_val; // XOR
                    default: wb_val = 32'd0;
                endcase
            end
            7'b0010011: begin // ADDI
                wb_en = (funct3 == 3'b000);
                wb_val = rs1_val + imm_i(instr);
            end
            7'b0000011: begin // LW
                mem_addr = rs1_val + imm_i(instr);
                mem_read = 1'b1;
                wb_en = 1'b1;
                wb_val = mem_rdata;
            end
            7'b0100011: begin // SW
                mem_addr = rs1_val + imm_s(instr);
                mem_wdata = rs2_val;
                mem_wstrb = 4'b1111;
                mem_write = 1'b1;
            end
            7'b1100011: begin // BEQ (funct3==000)
                if (funct3 == 3'b000 && rs1_val == rs2_val) pc_next = pc + imm_b(instr);
            end
            7'b1101111: begin // JAL
                wb_en = 1'b1;
                wb_val = pc + 32'd4;
                pc_next = pc + imm_j(instr);
            end
            7'b1100111: begin // JALR
                wb_en = 1'b1;
                wb_val = pc + 32'd4;
                pc_next = (rs1_val + imm_i(instr)) & ~32'd1;
            end
            default: ;
        endcase
    end

    integer r;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pc <= 32'h00000000;
            for (r = 0; r < 32; r = r + 1) regs[r] <= 32'd0;
            accel_start <= 1'b0;
            accel_base1 <= 32'd0;
            accel_base2 <= 32'd0;
            accel_len <= 32'd0;
        end else begin
            pc <= pc_next;
            if (wb_en && rd != 5'd0) regs[rd] <= wb_val;

            accel_start <= 1'b0;
            if (opcode == 7'b1111011 && funct3 == 3'b000) begin
                accel_start <= 1'b1;
                accel_base1 <= rs1_val;
                accel_base2 <= rs2_val;
                accel_len   <= {25'd0, funct7};
            end
        end
    end

endmodule
