using System.Text.Json;
using Sts2Emulator.Core.Run;
using Xunit;

namespace Sts2Emulator.Tests;

/// <summary>
/// Every act's map is its own size and its own mix, as the game builds it.
/// </summary>
/// <remarks>
/// <c>StandardActMap</c> is <c>GetNumberOfRooms() + 1</c> rows long, and
/// <c>BaseNumberOfRooms</c> is 15 for Overgrowth and Underdocks, 14 for Hive and 13 for
/// Glory, so the boss stands on row 16, 15 and 14. The treasure row is the boss row less
/// seven, the rest row the boss row less one, and no rest site may sit in the top three
/// rows -- all of it counted from the act's own length. Each act also rolls its own
/// <c>GetMapPointTypes</c>: Hive draws its rests around six and one Unknown fewer, and
/// Glory rolls rests with <c>NextInt(5, 7)</c>, also one Unknown fewer. The emulator gave
/// every act Overgrowth's seventeen rows and counts, so from act 2 on no real map could
/// be walked.
///
/// The expected maps are the game's own: every map screen of two real A2 runs that
/// reached acts 2 and 3 (spire-coach session logs), written to
/// <c>tests/fixtures/act_maps/real_act_maps.json</c>. A map is a function of the seed and
/// the act index alone, so nothing else about those runs has to be reproduced.
/// </remarks>
public class ActMapShapeTests
{
    private static readonly Dictionary<string, int> NodeTypes = new()
    {
        ["Monster"] = RunConstants.NodeNormal,
        ["Elite"] = RunConstants.NodeElite,
        ["RestSite"] = RunConstants.NodeRest,
        ["Shop"] = RunConstants.NodeShop,
        ["Treasure"] = RunConstants.NodeRelic,
        ["Boss"] = RunConstants.NodeBoss,
        ["Unknown"] = RunConstants.NodeEvent,
        ["Ancient"] = RunConstants.NodeAncient,
    };

    private static JsonElement Fixture()
    {
        string path = Path.Combine(
            AppContext.BaseDirectory,
            "..",
            "..",
            "..",
            "..",
            "..",
            "tests",
            "fixtures",
            "act_maps",
            "real_act_maps.json"
        );
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone();
    }

    public static IEnumerable<object[]> RealMaps()
    {
        foreach (var run in Fixture().GetProperty("runs").EnumerateObject())
        {
            foreach (var act in run.Value.GetProperty("acts").EnumerateObject())
            {
                // A Spoils Map carried into the act swaps in `SpoilsActMap`, an hourglass
                // drawn off its own "spoils_map" stream, which is not modelled yet.
                if (act.Value.TryGetProperty("spoils_map", out var spoils) && spoils.GetBoolean())
                {
                    continue;
                }

                yield return [run.Name, int.Parse(act.Name)];
            }
        }
    }

    [Theory]
    [MemberData(nameof(RealMaps))]
    public void TheActMapIsTheMapTheGameDrew(string seed, int act)
    {
        var run = Fixture().GetProperty("runs").GetProperty(seed);
        var expected = run.GetProperty("acts")
            .GetProperty(act.ToString())
            .GetProperty("nodes")
            .EnumerateArray()
            .Select(node =>
                (
                    Col: node[0].GetInt32(),
                    Row: node[1].GetInt32(),
                    Type: NodeTypes[node[2].GetString()!],
                    Children: string.Join(
                        " ",
                        node[3].EnumerateArray().Select(c => $"{c[0]},{c[1]}").OrderBy(c => c)
                    )
                )
            )
            .OrderBy(n => n.Row)
            .ThenBy(n => n.Col)
            .ToList();

        var engine = BossFights.InAct(seed, run.GetProperty("ascension").GetInt32(), act - 1);
        var actual = engine
            .State.MapNodes.Values.Where(n => n.Row > 0)
            .Select(n =>
                (
                    n.Col,
                    n.Row,
                    Type: n.NodeType,
                    Children: string.Join(
                        " ",
                        n.Children.Select(c => $"{c.Col},{c.Row}").OrderBy(c => c)
                    )
                )
            )
            .OrderBy(n => n.Row)
            .ThenBy(n => n.Col)
            .ToList();

        Assert.Equal(expected.Where(n => n.Row > 0), actual);
    }

    [Theory]
    [InlineData(RunConstants.ActOvergrowth, 16)]
    [InlineData(RunConstants.ActUnderdocks, 16)]
    [InlineData(RunConstants.ActHive, 15)]
    [InlineData(RunConstants.ActGlory, 14)]
    public void TheBossStandsOneRowPastTheActsRooms(int act, int bossRow)
    {
        Assert.Equal(bossRow, RunConstants.MapBossRowFor(act));
    }
}
