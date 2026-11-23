// SPDX-License-Identifier: Apache-2.0
// Ramec s vyvody pro BitTF mini na IHP SG13G2 (podle IHP-GmbH/ihp-sg13g2-librelane-template, Apache-2.0).
// 19 padu: napajeni 4 (VDD, VSS, IOVDD, IOVSS), vstupy 6 (clk, rst_n, host_own, spi_sck, spi_cs_n, spi_mosi),
// vystupy 3 (spi_miso, done, busy), obousmerne 6 (psram_sck, psram_cs_n, psram_io[3:0]).
`default_nettype none
module chip_top (
`ifdef USE_POWER_PINS
    inout wire IOVDD,
    inout wire IOVSS,
    inout wire VDD,
    inout wire VSS,
`endif
    inout wire clk_PAD,
    inout wire rst_n_PAD,
    inout wire [3:0] in_PAD,       // host_own, spi_sck, spi_cs_n, spi_mosi
    inout wire [2:0] out_PAD,      // spi_miso, done, busy
    inout wire [5:0] bidir_PAD     // psram_sck, psram_cs_n, psram_io[3:0]
);
    wire clk, rst_n;
    wire [3:0] in_c;
    wire [2:0] out_c;
    wire [5:0] bd_in, bd_out, bd_oe;

`ifdef USE_POWER_PINS
`define PWR .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
`define PWRC .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
`else
`define PWR
`define PWRC
`endif

    (* keep *) sg13g2_IOPadIOVdd iovdd_pad (`PWR);
    (* keep *) sg13g2_IOPadIOVss iovss_pad (`PWR);
    (* keep *) sg13g2_IOPadVdd   vdd_pad   (`PWR);
    (* keep *) sg13g2_IOPadVss   vss_pad   (`PWR);

    sg13g2_IOPadIn clk_pad   (`PWRC .p2c(clk),   .pad(clk_PAD));
    sg13g2_IOPadIn rst_n_pad (`PWRC .p2c(rst_n), .pad(rst_n_PAD));

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : inputs
            sg13g2_IOPadIn input_pad (`PWRC .p2c(in_c[i]), .pad(in_PAD[i]));
        end
        for (i = 0; i < 3; i = i + 1) begin : outputs
            sg13g2_IOPadOut16mA output_pad (`PWRC .c2p(out_c[i]), .pad(out_PAD[i]));
        end
        for (i = 0; i < 6; i = i + 1) begin : bidirs
            sg13g2_IOPadInOut16mA bidir_pad (`PWRC .c2p(bd_out[i]), .c2p_en(bd_oe[i]), .p2c(bd_in[i]), .pad(bidir_PAD[i]));
        end
    endgenerate

    wire       q_sck, q_cs_n, ctl_oe;
    wire [3:0] q_io_o, q_io_oe;
    assign bd_out = {q_io_o, q_cs_n, q_sck};
    assign bd_oe  = {q_io_oe, ctl_oe, ctl_oe};

    (* keep *) bittf_chip i_chip_core (
        .clk(clk), .rst_n(rst_n),
        .psram_sck(q_sck), .psram_cs_n(q_cs_n), .psram_io_o(q_io_o), .psram_io_oe(q_io_oe),
        .psram_ctl_oe(ctl_oe), .psram_io_i(bd_in[5:2]),
        .host_own(in_c[0]), .spi_sck(in_c[1]), .spi_cs_n(in_c[2]), .spi_mosi(in_c[3]),
        .spi_miso(out_c[0]), .done_o(out_c[1]), .busy_o(out_c[2])
    );
endmodule
`default_nettype wire
