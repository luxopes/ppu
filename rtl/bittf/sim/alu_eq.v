// Testovaci obal: puvodni bittf_alu vs bittf_mini_alu se stejnymi vstupy (soucin pro mini behavioralne).
module alu_eq (
    input  wire        clk, rst, start,
    input  wire [7:0]  fn,
    input  wire [2:0]  tc,
    input  wire        clamp_en, clamp_sym,
    input  wire [6:0]  cbits,
    input  wire [63:0] a, b, c, r1, r2,
    input  wire [31:0] imm, i0, i1, rnd,
    output wire        done_ref, done_mini,
    output wire [63:0] y_ref, y_mini
);
    wire        mneed;
    wire [63:0] ma, mb;
    bittf_alu ref_alu (.clk(clk), .rst(rst), .start(start), .fn(fn), .tc(tc), .clamp_en(clamp_en),
        .clamp_sym(clamp_sym), .cbits(cbits), .a(a), .b(b), .c(c), .r1(r1), .r2(r2), .imm(imm), .i0(i0),
        .i1(i1), .rnd(rnd), .done(done_ref), .y(y_ref));
    bittf_mini_alu mini_alu (.clk(clk), .rst(rst), .start(start), .fn(fn), .tc(tc), .clamp_en(clamp_en),
        .clamp_sym(clamp_sym), .cbits(cbits), .a(a), .b(b), .c(c), .r1(r1), .r2(r2), .imm(imm), .i0(i0),
        .i1(i1), .rnd(rnd), .mul_need(mneed), .mul_a(ma), .mul_b(mb), .mul_p(ma * mb),
        .done(done_mini), .y(y_mini));
endmodule
