// Postupna nasobicka: p = dolnich W bitu a*b (stejne pro znamenkova i bezznamenkova cisla).
// D bitu b za takt; konci drive, jakmile zbytek b je nulovy (male operandy = mene taktu).
// done = 1 takt po poslednim kroku, p plati od done do dalsiho start.
module bittf_mulseq #(
    parameter W = 64,
    parameter D = 8
) (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire [W-1:0] a,
    input  wire [W-1:0] b,
    output reg          done,
    output reg  [W-1:0] p
);
    reg [W-1:0] ar, br;
    reg busy;
    wire [W-1:0] part = ar * {{(W-D){1'b0}}, br[D-1:0]};
    wire [W-1:0] br_n = br >> D;

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start) begin
                ar <= a;
                br <= b;
                p <= {W{1'b0}};
                busy <= 1'b1;
            end else if (busy) begin
                p <= p + part;
                ar <= ar << D;
                br <= br_n;
                if (br_n == 0) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end
endmodule
